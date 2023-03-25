import torch
from torch import nn
from tasks.tts.ps_adv import PortaSpeechAdvTask, FastSpeechTask
from text_to_speech.utils.commons.hparams import hparams
from text_to_speech.utils.nn.seq_utils import group_hidden_by_segs


class PortaSpeechAdvMLMTask(PortaSpeechAdvTask):

    def build_scheduler(self, optimizer):
        return [
            FastSpeechTask.build_scheduler(self, optimizer[0]), # Generator Scheduler
            torch.optim.lr_scheduler.StepLR(optimizer=optimizer[1], # Discriminator Scheduler
                **hparams["discriminator_scheduler_params"]),
        ]

    def on_before_optimization(self, opt_idx):
        if opt_idx in [0, 2]:
            nn.utils.clip_grad_norm_(self.dp_params, hparams['clip_grad_norm'])
            if self.use_bert:
                nn.utils.clip_grad_norm_(self.bert_params, hparams['clip_grad_norm'])
                nn.utils.clip_grad_norm_(self.gen_params_except_bert_and_dp, hparams['clip_grad_norm'])
            else:
                nn.utils.clip_grad_norm_(self.gen_params_except_dp, hparams['clip_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.disc_params, hparams["clip_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler[0].step(self.global_step // hparams['accumulate_grad_batches'])
            self.scheduler[1].step(self.global_step // hparams['accumulate_grad_batches'])


    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output = {}
        loss_weights = {}
        disc_start = self.global_step >= hparams["disc_start_steps"] and hparams['lambda_mel_adv'] > 0
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            loss_output, model_out = self.run_model(sample, infer=False)
            self.model_out_gt = self.model_out = \
                {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
            if disc_start:
                mel_p = model_out['mel_out']
                if hasattr(self.model, 'out2mel'):
                    mel_p = self.model.out2mel(mel_p)
                o_ = self.mel_disc(mel_p)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    loss_output['a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                    loss_weights['a'] = hparams['lambda_mel_adv']
                if pc_ is not None:
                    loss_output['ac'] = self.mse_loss_fn(pc_, pc_.new_ones(pc_.size()))
                    loss_weights['ac'] = hparams['lambda_mel_adv']
            else:
                return None

            loss_output2, model_out2 = self.run_contrastive_learning(sample)
            loss_output.update(loss_output2)
            model_out.update(model_out2)
            
        elif optimizer_idx == 1:
            #######################
            #    Discriminator    #
            #######################
            if disc_start and self.global_step % hparams['disc_interval'] == 0:
                model_out = self.model_out_gt
                mel_g = sample['mels']
                mel_p = model_out['mel_out']
                o = self.mel_disc(mel_g)
                p, pc = o['y'], o['y_c']
                o_ = self.mel_disc(mel_p)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    loss_output["r"] = self.mse_loss_fn(p, p.new_ones(p.size()))
                    loss_output["f"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
                if pc_ is not None:
                    loss_output["rc"] = self.mse_loss_fn(pc, pc.new_ones(pc.size()))
                    loss_output["fc"] = self.mse_loss_fn(pc_, pc_.new_zeros(pc_.size()))

        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['txt_tokens'].size()[0]
        return total_loss, loss_output

    def run_contrastive_learning(self, sample):
        losses = {}
        outputs = {}

        bert = self.model.encoder.bert.bert
        bert_for_mlm = self.model.encoder.bert
        pooler = self.model.encoder.pooler
        sim = self.model.encoder.sim
        tokenizer = self.model.encoder.tokenizer
        ph_encoder = self.model.encoder

        if hparams['lambda_cl'] > 0:
            if hparams.get("cl_version", "v1") == "v1":
                cl_feats = sample['cl_feats']
                bs, _, t = cl_feats['cl_input_ids'].shape
                cl_input_ids = cl_feats['cl_input_ids'].reshape([bs*2, t])
                cl_attention_mask = cl_feats['cl_attention_mask'].reshape([bs*2, t])
                cl_token_type_ids = cl_feats['cl_token_type_ids'].reshape([bs*2, t])
                cl_output = bert(cl_input_ids, attention_mask=cl_attention_mask,token_type_ids=cl_token_type_ids,)
                pooler_output = pooler(cl_attention_mask, cl_output)
                pooler_output = pooler_output.reshape([bs, 2, -1])
                z1, z2 = pooler_output[:,0], pooler_output[:,1]

                cos_sim = sim(z1.unsqueeze(1), z2.unsqueeze(0))
                labels = torch.arange(cos_sim.size(0)).long().to(z1.device)
                ce_fn = nn.CrossEntropyLoss()
                cl_loss = ce_fn(cos_sim, labels)
                losses['cl_v'] = cl_loss.detach()
                losses['cl'] = cl_loss * hparams['lambda_cl']
            elif hparams['cl_version'] == "v2":
                # use the output of ph encoder as sentence embedding
                cl_feats = sample['cl_feats']
                bs, _, t = cl_feats['cl_input_ids'].shape
                cl_input_ids = cl_feats['cl_input_ids'].reshape([bs*2, t])
                cl_attention_mask = cl_feats['cl_attention_mask'].reshape([bs*2, t])
                cl_token_type_ids = cl_feats['cl_token_type_ids'].reshape([bs*2, t])
                txt_tokens = sample['txt_tokens']
                bert_feats = sample['bert_feats']
                src_nonpadding = (txt_tokens > 0).float()[:, :, None]
                ph_encoder_out1 = ph_encoder(txt_tokens, bert_feats=bert_feats, ph2word=sample['ph2word']) * src_nonpadding
                ph_encoder_out2 = ph_encoder(txt_tokens, bert_feats=bert_feats, ph2word=sample['ph2word']) * src_nonpadding
                # word_encoding1 = group_hidden_by_segs(ph_encoder_out1, sample['ph2word'], sample['ph2word'].max().item())
                # word_encoding2 = group_hidden_by_segs(ph_encoder_out2, sample['ph2word'], sample['ph2word'].max().item())
                z1 = ((ph_encoder_out1 * src_nonpadding).sum(1) / src_nonpadding.sum(1))
                z2 = ((ph_encoder_out2 * src_nonpadding).sum(1) / src_nonpadding.sum(1))

                cos_sim = sim(z1.unsqueeze(1), z2.unsqueeze(0))
                labels = torch.arange(cos_sim.size(0)).long().to(z1.device)
                ce_fn = nn.CrossEntropyLoss()
                cl_loss = ce_fn(cos_sim, labels)
                losses['cl_v'] = cl_loss.detach()
                losses['cl'] = cl_loss * hparams['lambda_cl']
            elif hparams['cl_version'] == "v3":
                # use the word-level contrastive learning
                cl_feats = sample['cl_feats']
                bs, _, t = cl_feats['cl_input_ids'].shape
                cl_input_ids = cl_feats['cl_input_ids'].reshape([bs*2, t])
                cl_attention_mask = cl_feats['cl_attention_mask'].reshape([bs*2, t])
                cl_token_type_ids = cl_feats['cl_token_type_ids'].reshape([bs*2, t])
                cl_output = bert(cl_input_ids, attention_mask=cl_attention_mask,token_type_ids=cl_token_type_ids,)
                cl_output = cl_output.last_hidden_state.reshape([-1, 768]) # [bs*2,t_w,768] ==> [bs*2*t_w, 768]
                cl_word_out = cl_output[cl_attention_mask.reshape([-1]).bool()] # [num_word*2, 768]
                cl_word_out = cl_word_out.view([-1, 2, 768])
                z1_total, z2_total = cl_word_out[:,0], cl_word_out[:,1] # [num_word, 768]
                ce_fn = nn.CrossEntropyLoss()
                start_idx = 0
                lengths = cl_attention_mask.sum(-1)
                cl_loss_accu = 0
                for i in range(bs):
                    length = lengths[i]
                    z1 = z1_total[start_idx:start_idx + length]
                    z2 = z2_total[start_idx:start_idx + length]
                    start_idx += length
                    cos_sim = sim(z1.unsqueeze(1), z2.unsqueeze(0))
                    labels = torch.arange(cos_sim.size(0)).long().to(z1.device)
                    cl_loss_accu += ce_fn(cos_sim, labels) * length
                cl_loss = cl_loss_accu / lengths.sum()
                losses['cl_v'] = cl_loss.detach()
                losses['cl'] = cl_loss * hparams['lambda_cl']
            elif hparams['cl_version'] == "v4":
                # with Wiki dataset
                cl_feats = sample['cl_feats']
                bs, _, t = cl_feats['cl_input_ids'].shape
                cl_input_ids = cl_feats['cl_input_ids'].reshape([bs*2, t])
                cl_attention_mask = cl_feats['cl_attention_mask'].reshape([bs*2, t])
                cl_token_type_ids = cl_feats['cl_token_type_ids'].reshape([bs*2, t])
                cl_output = bert(cl_input_ids, attention_mask=cl_attention_mask,token_type_ids=cl_token_type_ids,)
                pooler_output = pooler(cl_attention_mask, cl_output)
                pooler_output = pooler_output.reshape([bs, 2, -1])
                z1, z2 = pooler_output[:,0], pooler_output[:,1]

                cos_sim = sim(z1.unsqueeze(1), z2.unsqueeze(0))
                labels = torch.arange(cos_sim.size(0)).long().to(z1.device)
                ce_fn = nn.CrossEntropyLoss()
                cl_loss = ce_fn(cos_sim, labels)
                losses['cl_v'] = cl_loss.detach()
                losses['cl'] = cl_loss * hparams['lambda_cl']
            elif hparams['cl_version'] == "v5":
                # with NLI dataset
                cl_feats = sample['cl_feats']
                cl_input_ids = cl_feats['sent0']['cl_input_ids']
                cl_attention_mask = cl_feats['sent0']['cl_attention_mask']
                cl_token_type_ids = cl_feats['sent0']['cl_token_type_ids']
                cl_output = bert(cl_input_ids, attention_mask=cl_attention_mask,token_type_ids=cl_token_type_ids,)
                z1 = pooler_output_sent0 = pooler(cl_attention_mask, cl_output)

                cl_input_ids = cl_feats['sent1']['cl_input_ids']
                cl_attention_mask = cl_feats['sent1']['cl_attention_mask']
                cl_token_type_ids = cl_feats['sent1']['cl_token_type_ids']
                cl_output = bert(cl_input_ids, attention_mask=cl_attention_mask,token_type_ids=cl_token_type_ids,)
                z2 = pooler_output_sent1 = pooler(cl_attention_mask, cl_output)

                cl_input_ids = cl_feats['hard_neg']['cl_input_ids']
                cl_attention_mask = cl_feats['hard_neg']['cl_attention_mask']
                cl_token_type_ids = cl_feats['hard_neg']['cl_token_type_ids']
                cl_output = bert(cl_input_ids, attention_mask=cl_attention_mask,token_type_ids=cl_token_type_ids,)
                z3 = pooler_output_neg = pooler(cl_attention_mask, cl_output)

                cos_sim = sim(z1.unsqueeze(1), z2.unsqueeze(0))
                z1_z3_cos = sim(z1.unsqueeze(1), z3.unsqueeze(0))
                cos_sim = torch.cat([cos_sim, z1_z3_cos], 1) # [n_sent, n_sent * 2]
                labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device) # [n_sent, ]
                ce_fn = nn.CrossEntropyLoss()
                cl_loss = ce_fn(cos_sim, labels)
                losses['cl_v'] = cl_loss.detach()
                losses['cl'] = cl_loss * hparams['lambda_cl']
            else:
                raise NotImplementedError()

        if hparams['lambda_mlm'] > 0:
            cl_feats = sample['cl_feats']
            mlm_input_ids = cl_feats['mlm_input_ids']
            bs, t = mlm_input_ids.shape
            mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
            mlm_labels = cl_feats['mlm_labels']
            mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
            mlm_attention_mask = cl_feats['mlm_attention_mask']

            prediction_scores = bert_for_mlm(mlm_input_ids, mlm_attention_mask).logits
            ce_fn = nn.CrossEntropyLoss(reduction="none")
            mlm_loss = ce_fn(prediction_scores.view(-1, tokenizer.vocab_size), mlm_labels.view(-1))
            mlm_loss = mlm_loss[mlm_labels.view(-1)>=0].mean()
            losses['mlm'] = mlm_loss * hparams['lambda_mlm']
            losses['mlm_v'] = mlm_loss.detach()

        return losses, outputs
        