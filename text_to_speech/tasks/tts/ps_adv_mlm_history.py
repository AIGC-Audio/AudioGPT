import torch
from torch import nn
from tasks.tts.ps_adv import PortaSpeechAdvTask, FastSpeechTask
from text_to_speech.utils.commons.hparams import hparams


class PortaSpeechAdvMLMTask(PortaSpeechAdvTask):

    def build_optimizer(self, model):
        optimizer_gen = torch.optim.AdamW(
            self.model.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])

        optimizer_disc = torch.optim.AdamW(
            self.disc_params,
            lr=hparams['disc_lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            **hparams["discriminator_optimizer_params"]) if len(self.disc_params) > 0 else None

        optimizer_encoder = torch.optim.AdamW(
            self.model.encoder.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        return [optimizer_gen, optimizer_disc, optimizer_encoder]

    def build_scheduler(self, optimizer):
        return [
            FastSpeechTask.build_scheduler(self, optimizer[0]), # Generator Scheduler
            torch.optim.lr_scheduler.StepLR(optimizer=optimizer[1], # Discriminator Scheduler
                **hparams["discriminator_scheduler_params"]),
            FastSpeechTask.build_scheduler(self, optimizer[2]), # Generator Scheduler
        ]

    def on_before_optimization(self, opt_idx):
        if opt_idx in [0,2]:
            nn.utils.clip_grad_norm_(self.dp_params, hparams['clip_grad_norm'])
            if self.use_graph_encoder:
                nn.utils.clip_grad_norm_(self.gen_params_except_gae_and_dp, hparams['clip_grad_norm'])
                nn.utils.clip_grad_norm_(self.gae_params, hparams['clip_grad_norm'])
            elif self.use_bert:
                nn.utils.clip_grad_norm_(self.gen_params_except_bert_and_dp, hparams['clip_grad_norm'])
                nn.utils.clip_grad_norm_(self.bert_params, hparams['clip_grad_norm'])
            else:
                nn.utils.clip_grad_norm_(self.gen_params_except_dp, hparams['clip_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.disc_params, hparams["clip_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler[0].step(self.global_step // hparams['accumulate_grad_batches'])
            self.scheduler[1].step(self.global_step // hparams['accumulate_grad_batches'])
            self.scheduler[2].step(self.global_step // hparams['accumulate_grad_batches'])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler[0].step(self.global_step // hparams['accumulate_grad_batches'])
            self.scheduler[1].step(self.global_step // hparams['accumulate_grad_batches'])
            self.scheduler[2].step(self.global_step // hparams['accumulate_grad_batches'])

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
        else:
            loss_output, model_out = self.run_contrastive_learning(sample)

        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['txt_tokens'].size()[0]
        return total_loss, loss_output

    def run_contrastive_learning(self, sample):
        losses = {}
        outputs = {}

        bert = self.model.encoder.bert
        pooler = self.model.encoder.pooler
        sim = self.model.encoder.sim
        # electra_gen = self.model.encoder.electra_gen
        # electra_disc = self.model.encoder.electra_disc
        # electra_head = self.model.encoder.electra_head
        
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
        losses['cl'] = cl_loss * hparams['lambda_mlm']

        # mlm_input_ids = cl_feats['mlm_input_ids']
        # mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        # with torch.no_grad():
        #     g_pred = electra_gen(mlm_input_ids, cl_attention_mask)[0].argmax(-1)
        # g_pred[:, 0] = 101 # CLS token
        # replaced = (g_pred != cl_input_ids) * cl_attention_mask
        # e_inputs = g_pred * cl_attention_mask
        # mlm_outputs = electra_disc(
        #     e_inputs,
        #     attention_mask=cl_attention_mask,
        #     token_type_ids=cl_token_type_ids,
        #     position_ids=None,
        #     head_mask=None,
        #     inputs_embeds=None,
        #     output_attentions=None,
        #     output_hidden_states=False, # True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        #     return_dict=True,
        #     cls_input=pooler_output.view((-1, pooler_output.size(-1))),
        # )
        # e_labels = replaced.view(-1, replaced.size(-1))
        # prediction_scores = electra_head(mlm_outputs.last_hidden_state)
        # # rep = (e_labels == 1) * cl_attention_mask
        # # fix = (e_labels == 0) * cl_attention_mask
        # # prediction = prediction_scores.argmax(-1)
        # # self.electra_rep_acc = float((prediction*rep).sum()/rep.sum())
        # # self.electra_fix_acc = float(1.0 - (prediction*fix).sum()/fix.sum())
        # # self.electra_acc = float(((prediction == e_labels) * cl_attention_mask).sum()/cl_attention_mask.sum())
        # masked_lm_loss = ce_fn(prediction_scores.view(-1, 2), e_labels.view(-1))
        # losses['mlm_v'] = masked_lm_loss.detach()
        # losses['mlm'] = masked_lm_loss * hparams['lambda_mlm']

        return losses, outputs
        