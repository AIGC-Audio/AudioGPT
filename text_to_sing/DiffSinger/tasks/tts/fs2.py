import matplotlib

matplotlib.use('Agg')

from utils import audio
import matplotlib.pyplot as plt
from data_gen.tts.data_gen_utils import get_pitch
from tasks.tts.fs2_utils import FastSpeechDataset
from utils.cwt import cwt2f0
from utils.pl_utils import data_loader
import os
from multiprocessing.pool import Pool
from tqdm import tqdm
from modules.fastspeech.tts_modules import mel2ph_to_dur
from utils.hparams import hparams
from utils.plot import spec_to_figure, dur_to_figure, f0_to_figure
from utils.pitch_utils import denorm_f0
from modules.fastspeech.fs2 import FastSpeech2
from tasks.tts.tts import TtsTask
import torch
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import utils
import torch.distributions
import numpy as np
from modules.commons.ssim import ssim

class FastSpeech2Task(TtsTask):
    def __init__(self):
        super(FastSpeech2Task, self).__init__()
        self.dataset_cls = FastSpeechDataset
        self.mse_loss_fn = torch.nn.MSELoss()
        mel_losses = hparams['mel_loss'].split("|")
        self.loss_and_lambda = {}
        for i, l in enumerate(mel_losses):
            if l == '':
                continue
            if ':' in l:
                l, lbd = l.split(":")
                lbd = float(lbd)
            else:
                lbd = 1.0
            self.loss_and_lambda[l] = lbd
        print("| Mel losses:", self.loss_and_lambda)
        self.sil_ph = self.phone_encoder.sil_phonemes()

    @data_loader
    def train_dataloader(self):
        train_dataset = self.dataset_cls(hparams['train_set_name'], shuffle=True)
        return self.build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences,
                                     endless=hparams['endless_ds'])

    @data_loader
    def val_dataloader(self):
        valid_dataset = self.dataset_cls(hparams['valid_set_name'], shuffle=False)
        return self.build_dataloader(valid_dataset, False, self.max_eval_tokens, self.max_eval_sentences)

    @data_loader
    def test_dataloader(self):
        test_dataset = self.dataset_cls(hparams['test_set_name'], shuffle=False)
        return self.build_dataloader(test_dataset, False, self.max_eval_tokens,
                                     self.max_eval_sentences, batch_by_size=False)

    def build_tts_model(self):
        self.model = FastSpeech2(self.phone_encoder)

    def build_model(self):
        self.build_tts_model()
        if hparams['load_ckpt'] != '':
            self.load_ckpt(hparams['load_ckpt'], strict=True)
        utils.print_arch(self.model)
        return self.model

    def _training_step(self, sample, batch_idx, _):
        loss_output = self.run_model(self.model, sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['txt_tokens'].size()[0]
        return total_loss, loss_output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        mel_out = self.model.out2mel(model_out['mel_out'])
        outputs = utils.tensors_to_scalars(outputs)
        # if sample['mels'].shape[0] == 1:
        #     self.add_laplace_var(mel_out, sample['mels'], outputs)
        if batch_idx < hparams['num_valid_plots']:
            self.plot_mel(batch_idx, sample['mels'], mel_out)
            self.plot_dur(batch_idx, sample, model_out)
            if hparams['use_pitch_embed']:
                self.plot_pitch(batch_idx, sample, model_out)
        return outputs

    def _validation_end(self, outputs):
        all_losses_meter = {
            'total_loss': utils.AvgrageMeter(),
        }
        for output in outputs:
            n = output['nsamples']
            for k, v in output['losses'].items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = utils.AvgrageMeter()
                all_losses_meter[k].update(v, n)
            all_losses_meter['total_loss'].update(output['total_loss'], n)
        return {k: round(v.avg, 4) for k, v in all_losses_meter.items()}

    def run_model(self, model, sample, return_output=False):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        energy = sample['energy']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        if hparams['pitch_type'] == 'cwt':
            cwt_spec = sample[f'cwt_spec']
            f0_mean = sample['f0_mean']
            f0_std = sample['f0_std']
            sample['f0_cwt'] = f0 = model.cwt2f0_norm(cwt_spec, f0_mean, f0_std, mel2ph)

        output = model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed,
                       ref_mels=target, f0=f0, uv=uv, energy=energy, infer=False)

        losses = {}
        self.add_mel_loss(output['mel_out'], target, losses)
        self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
        if hparams['use_pitch_embed']:
            self.add_pitch_loss(output, sample, losses)
        if hparams['use_energy_embed']:
            self.add_energy_loss(output['energy_pred'], energy, losses)
        if not return_output:
            return losses
        else:
            return losses, output

    ############
    # losses
    ############
    def add_mel_loss(self, mel_out, target, losses, postfix='', mel_mix_loss=None):
        if mel_mix_loss is None:
            for loss_name, lbd in self.loss_and_lambda.items():
                if 'l1' == loss_name:
                    l = self.l1_loss(mel_out, target)
                elif 'mse' == loss_name:
                    raise NotImplementedError
                elif 'ssim' == loss_name:
                    l = self.ssim_loss(mel_out, target)
                elif 'gdl' == loss_name:
                    raise NotImplementedError
                losses[f'{loss_name}{postfix}'] = l * lbd
        else:
            raise NotImplementedError

    def l1_loss(self, decoder_output, target):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        l1_loss = F.l1_loss(decoder_output, target, reduction='none')
        weights = self.weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss

    def ssim_loss(self, decoder_output, target, bias=6.0):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        assert decoder_output.shape == target.shape
        weights = self.weights_nonzero_speech(target)
        decoder_output = decoder_output[:, None] + bias
        target = target[:, None] + bias
        ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
        ssim_loss = (ssim_loss * weights).sum() / weights.sum()
        return ssim_loss

    def add_dur_loss(self, dur_pred, mel2ph, txt_tokens, losses=None):
        """

        :param dur_pred: [B, T], float, log scale
        :param mel2ph: [B, T]
        :param txt_tokens: [B, T]
        :param losses:
        :return:
        """
        B, T = txt_tokens.shape
        nonpadding = (txt_tokens != 0).float()
        dur_gt = mel2ph_to_dur(mel2ph, T).float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p in self.sil_ph:
            is_sil = is_sil | (txt_tokens == self.phone_encoder.encode(p)[0])
        is_sil = is_sil.float()  # [B, T_txt]

        # phone duration loss
        if hparams['dur_loss'] == 'mse':
            losses['pdur'] = F.mse_loss(dur_pred, (dur_gt + 1).log(), reduction='none')
            losses['pdur'] = (losses['pdur'] * nonpadding).sum() / nonpadding.sum()
            dur_pred = (dur_pred.exp() - 1).clamp(min=0)
        elif hparams['dur_loss'] == 'mog':
            return NotImplementedError
        elif hparams['dur_loss'] == 'crf':
            losses['pdur'] = -self.model.dur_predictor.crf(
                dur_pred, dur_gt.long().clamp(min=0, max=31), mask=nonpadding > 0, reduction='mean')
        losses['pdur'] = losses['pdur'] * hparams['lambda_ph_dur']

        # use linear scale for sent and word duration
        if hparams['lambda_word_dur'] > 0:
            word_id = (is_sil.cumsum(-1) * (1 - is_sil)).long()
            word_dur_p = dur_pred.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_pred)[:, 1:]
            word_dur_g = dur_gt.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_gt)[:, 1:]
            wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction='none')
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses['wdur'] = wdur_loss * hparams['lambda_word_dur']
        if hparams['lambda_sent_dur'] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction='mean')
            losses['sdur'] = sdur_loss.mean() * hparams['lambda_sent_dur']

    def add_pitch_loss(self, output, sample, losses):
        if hparams['pitch_type'] == 'ph':
            nonpadding = (sample['txt_tokens'] != 0).float()
            pitch_loss_fn = F.l1_loss if hparams['pitch_loss'] == 'l1' else F.mse_loss
            losses['f0'] = (pitch_loss_fn(output['pitch_pred'][:, :, 0], sample['f0'],
                                          reduction='none') * nonpadding).sum() \
                           / nonpadding.sum() * hparams['lambda_f0']
            return
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (mel2ph != 0).float()
        if hparams['pitch_type'] == 'cwt':
            cwt_spec = sample[f'cwt_spec']
            f0_mean = sample['f0_mean']
            f0_std = sample['f0_std']
            cwt_pred = output['cwt'][:, :, :10]
            f0_mean_pred = output['f0_mean']
            f0_std_pred = output['f0_std']
            losses['C'] = self.cwt_loss(cwt_pred, cwt_spec) * hparams['lambda_f0']
            if hparams['use_uv']:
                assert output['cwt'].shape[-1] == 11
                uv_pred = output['cwt'][:, :, -1]
                losses['uv'] = (F.binary_cross_entropy_with_logits(uv_pred, uv, reduction='none') * nonpadding) \
                                   .sum() / nonpadding.sum() * hparams['lambda_uv']
            losses['f0_mean'] = F.l1_loss(f0_mean_pred, f0_mean) * hparams['lambda_f0']
            losses['f0_std'] = F.l1_loss(f0_std_pred, f0_std) * hparams['lambda_f0']
            if hparams['cwt_add_f0_loss']:
                f0_cwt_ = self.model.cwt2f0_norm(cwt_pred, f0_mean_pred, f0_std_pred, mel2ph)
                self.add_f0_loss(f0_cwt_[:, :, None], f0, uv, losses, nonpadding=nonpadding)
        elif hparams['pitch_type'] == 'frame':
            self.add_f0_loss(output['pitch_pred'], f0, uv, losses, nonpadding=nonpadding)

    def add_f0_loss(self, p_pred, f0, uv, losses, nonpadding):
        assert p_pred[..., 0].shape == f0.shape
        if hparams['use_uv']:
            assert p_pred[..., 1].shape == uv.shape
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1], uv, reduction='none') * nonpadding).sum() \
                           / nonpadding.sum() * hparams['lambda_uv']
            nonpadding = nonpadding * (uv == 0).float()

        f0_pred = p_pred[:, :, 0]
        if hparams['pitch_loss'] in ['l1', 'l2']:
            pitch_loss_fn = F.l1_loss if hparams['pitch_loss'] == 'l1' else F.mse_loss
            losses['f0'] = (pitch_loss_fn(f0_pred, f0, reduction='none') * nonpadding).sum() \
                           / nonpadding.sum() * hparams['lambda_f0']
        elif hparams['pitch_loss'] == 'ssim':
            return NotImplementedError

    def cwt_loss(self, cwt_p, cwt_g):
        if hparams['cwt_loss'] == 'l1':
            return F.l1_loss(cwt_p, cwt_g)
        if hparams['cwt_loss'] == 'l2':
            return F.mse_loss(cwt_p, cwt_g)
        if hparams['cwt_loss'] == 'ssim':
            return self.ssim_loss(cwt_p, cwt_g, 20)

    def add_energy_loss(self, energy_pred, energy, losses):
        nonpadding = (energy != 0).float()
        loss = (F.mse_loss(energy_pred, energy, reduction='none') * nonpadding).sum() / nonpadding.sum()
        loss = loss * hparams['lambda_energy']
        losses['e'] = loss


    ############
    # validation plots
    ############
    def plot_mel(self, batch_idx, spec, spec_out, name=None):
        spec_cat = torch.cat([spec, spec_out], -1)
        name = f'mel_{batch_idx}' if name is None else name
        vmin = hparams['mel_vmin']
        vmax = hparams['mel_vmax']
        self.logger.experiment.add_figure(name, spec_to_figure(spec_cat[0], vmin, vmax), self.global_step)

    def plot_dur(self, batch_idx, sample, model_out):
        T_txt = sample['txt_tokens'].shape[1]
        dur_gt = mel2ph_to_dur(sample['mel2ph'], T_txt)[0]
        dur_pred = self.model.dur_predictor.out2dur(model_out['dur']).float()
        txt = self.phone_encoder.decode(sample['txt_tokens'][0].cpu().numpy())
        txt = txt.split(" ")
        self.logger.experiment.add_figure(
            f'dur_{batch_idx}', dur_to_figure(dur_gt, dur_pred, txt), self.global_step)

    def plot_pitch(self, batch_idx, sample, model_out):
        f0 = sample['f0']
        if hparams['pitch_type'] == 'ph':
            mel2ph = sample['mel2ph']
            f0 = self.expand_f0_ph(f0, mel2ph)
            f0_pred = self.expand_f0_ph(model_out['pitch_pred'][:, :, 0], mel2ph)
            self.logger.experiment.add_figure(
                f'f0_{batch_idx}', f0_to_figure(f0[0], None, f0_pred[0]), self.global_step)
            return
        f0 = denorm_f0(f0, sample['uv'], hparams)
        if hparams['pitch_type'] == 'cwt':
            # cwt
            cwt_out = model_out['cwt']
            cwt_spec = cwt_out[:, :, :10]
            cwt = torch.cat([cwt_spec, sample['cwt_spec']], -1)
            self.logger.experiment.add_figure(f'cwt_{batch_idx}', spec_to_figure(cwt[0]), self.global_step)
            # f0
            f0_pred = cwt2f0(cwt_spec, model_out['f0_mean'], model_out['f0_std'], hparams['cwt_scales'])
            if hparams['use_uv']:
                assert cwt_out.shape[-1] == 11
                uv_pred = cwt_out[:, :, -1] > 0
                f0_pred[uv_pred > 0] = 0
            f0_cwt = denorm_f0(sample['f0_cwt'], sample['uv'], hparams)
            self.logger.experiment.add_figure(
                f'f0_{batch_idx}', f0_to_figure(f0[0], f0_cwt[0], f0_pred[0]), self.global_step)
        elif hparams['pitch_type'] == 'frame':
            # f0
            uv_pred = model_out['pitch_pred'][:, :, 1] > 0
            pitch_pred = denorm_f0(model_out['pitch_pred'][:, :, 0], uv_pred, hparams)
            self.logger.experiment.add_figure(
                f'f0_{batch_idx}', f0_to_figure(f0[0], None, pitch_pred[0]), self.global_step)

    ############
    # infer
    ############
    def test_step(self, sample, batch_idx):
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        txt_tokens = sample['txt_tokens']
        mel2ph, uv, f0 = None, None, None
        ref_mels = None
        if hparams['profile_infer']:
            pass
        else:
            if hparams['use_gt_dur']:
                mel2ph = sample['mel2ph']
            if hparams['use_gt_f0']:
                f0 = sample['f0']
                uv = sample['uv']
                print('Here using gt f0!!')
            if hparams.get('use_midi') is not None and hparams['use_midi']:
                outputs = self.model(
                    txt_tokens, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, ref_mels=ref_mels, infer=True,
                    pitch_midi=sample['pitch_midi'], midi_dur=sample.get('midi_dur'), is_slur=sample.get('is_slur'))
            else:
                outputs = self.model(
                    txt_tokens, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, ref_mels=ref_mels, infer=True)
            sample['outputs'] = self.model.out2mel(outputs['mel_out'])
            sample['mel2ph_pred'] = outputs['mel2ph']
            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                sample['f0'] = self.pe(sample['mels'])['f0_denorm_pred']  # pe predict from GT mel
                sample['f0_pred'] = self.pe(sample['outputs'])['f0_denorm_pred']  # pe predict from Pred mel
            else:
                sample['f0'] = denorm_f0(sample['f0'], sample['uv'], hparams)
                sample['f0_pred'] = outputs.get('f0_denorm')
            return self.after_infer(sample)

    def after_infer(self, predictions):
        if self.saving_result_pool is None and not hparams['profile_infer']:
            self.saving_result_pool = Pool(min(int(os.getenv('N_PROC', os.cpu_count())), 16))
            self.saving_results_futures = []
        predictions = utils.unpack_dict_to_list(predictions)
        t = tqdm(predictions)
        for num_predictions, prediction in enumerate(t):
            for k, v in prediction.items():
                if type(v) is torch.Tensor:
                    prediction[k] = v.cpu().numpy()

            item_name = prediction.get('item_name')
            text = prediction.get('text').replace(":", "%3A")[:80]

            # remove paddings
            mel_gt = prediction["mels"]
            mel_gt_mask = np.abs(mel_gt).sum(-1) > 0
            mel_gt = mel_gt[mel_gt_mask]
            mel2ph_gt = prediction.get("mel2ph")
            mel2ph_gt = mel2ph_gt[mel_gt_mask] if mel2ph_gt is not None else None
            mel_pred = prediction["outputs"]
            mel_pred_mask = np.abs(mel_pred).sum(-1) > 0
            mel_pred = mel_pred[mel_pred_mask]
            mel_gt = np.clip(mel_gt, hparams['mel_vmin'], hparams['mel_vmax'])
            mel_pred = np.clip(mel_pred, hparams['mel_vmin'], hparams['mel_vmax'])

            mel2ph_pred = prediction.get("mel2ph_pred")
            if mel2ph_pred is not None:
                if len(mel2ph_pred) > len(mel_pred_mask):
                    mel2ph_pred = mel2ph_pred[:len(mel_pred_mask)]
                mel2ph_pred = mel2ph_pred[mel_pred_mask]

            f0_gt = prediction.get("f0")
            f0_pred = prediction.get("f0_pred")
            if f0_pred is not None:
                f0_gt = f0_gt[mel_gt_mask]
                if len(f0_pred) > len(mel_pred_mask):
                    f0_pred = f0_pred[:len(mel_pred_mask)]
                f0_pred = f0_pred[mel_pred_mask]

            str_phs = None
            if self.phone_encoder is not None and 'txt_tokens' in prediction:
                str_phs = self.phone_encoder.decode(prediction['txt_tokens'], strip_padding=True)
            gen_dir = os.path.join(hparams['work_dir'],
                                   f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
            wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)
            if not hparams['profile_infer']:
                os.makedirs(gen_dir, exist_ok=True)
                os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
                os.makedirs(f'{gen_dir}/plot', exist_ok=True)
                os.makedirs(os.path.join(hparams['work_dir'], 'P_mels_npy'), exist_ok=True)
                os.makedirs(os.path.join(hparams['work_dir'], 'G_mels_npy'), exist_ok=True)
                self.saving_results_futures.append(
                    self.saving_result_pool.apply_async(self.save_result, args=[
                        wav_pred, mel_pred, 'P', item_name, text, gen_dir, str_phs, mel2ph_pred, f0_gt, f0_pred]))

                if mel_gt is not None and hparams['save_gt']:
                    wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
                    self.saving_results_futures.append(
                        self.saving_result_pool.apply_async(self.save_result, args=[
                            wav_gt, mel_gt, 'G', item_name, text, gen_dir, str_phs, mel2ph_gt, f0_gt, f0_pred]))
                    if hparams['save_f0']:
                        import matplotlib.pyplot as plt
                        # f0_pred_, _ = get_pitch(wav_pred, mel_pred, hparams)
                        f0_pred_ = f0_pred
                        f0_gt_, _ = get_pitch(wav_gt, mel_gt, hparams)
                        fig = plt.figure()
                        plt.plot(f0_pred_, label=r'$f0_P$')
                        plt.plot(f0_gt_, label=r'$f0_G$')
                        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                            # f0_midi = prediction.get("f0_midi")
                            # f0_midi = f0_midi[mel_gt_mask]
                            # plt.plot(f0_midi, label=r'$f0_M$')
                            pass
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(f'{gen_dir}/plot/[F0][{item_name}]{text}.png', format='png')
                        plt.close(fig)

                t.set_description(
                    f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
            else:
                if 'gen_wav_time' not in self.stats:
                    self.stats['gen_wav_time'] = 0
                self.stats['gen_wav_time'] += len(wav_pred) / hparams['audio_sample_rate']
                print('gen_wav_time: ', self.stats['gen_wav_time'])

        return {}

    @staticmethod
    def save_result(wav_out, mel, prefix, item_name, text, gen_dir, str_phs=None, mel2ph=None, gt_f0=None, pred_f0=None):
        item_name = item_name.replace('/', '-')
        base_fn = f'[{item_name}][{prefix}]'

        if text is not None:
            base_fn += text
        base_fn += ('-' + hparams['exp_name'])
        np.save(os.path.join(hparams['work_dir'], f'{prefix}_mels_npy', item_name), mel)
        audio.save_wav(wav_out, f'{gen_dir}/wavs/{base_fn}.wav', hparams['audio_sample_rate'],
                       norm=hparams['out_wav_norm'])
        fig = plt.figure(figsize=(14, 10))
        spec_vmin = hparams['mel_vmin']
        spec_vmax = hparams['mel_vmax']
        heatmap = plt.pcolor(mel.T, vmin=spec_vmin, vmax=spec_vmax)
        fig.colorbar(heatmap)
        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
            gt_f0 = (gt_f0 - 100) / (800 - 100) * 80 * (gt_f0 > 0)
            pred_f0 = (pred_f0 - 100) / (800 - 100) * 80 * (pred_f0 > 0)
            plt.plot(pred_f0, c='white', linewidth=1, alpha=0.6)
            plt.plot(gt_f0, c='red', linewidth=1, alpha=0.6)
        else:
            f0, _ = get_pitch(wav_out, mel, hparams)
            f0 = (f0 - 100) / (800 - 100) * 80 * (f0 > 0)
            plt.plot(f0, c='white', linewidth=1, alpha=0.6)
        if mel2ph is not None and str_phs is not None:
            decoded_txt = str_phs.split(" ")
            dur = mel2ph_to_dur(torch.LongTensor(mel2ph)[None, :], len(decoded_txt))[0].numpy()
            dur = [0] + list(np.cumsum(dur))
            for i in range(len(dur) - 1):
                shift = (i % 20) + 1
                plt.text(dur[i], shift, decoded_txt[i])
                plt.hlines(shift, dur[i], dur[i + 1], colors='b' if decoded_txt[i] != '|' else 'black')
                plt.vlines(dur[i], 0, 5, colors='b' if decoded_txt[i] != '|' else 'black',
                           alpha=1, linewidth=1)
        plt.tight_layout()
        plt.savefig(f'{gen_dir}/plot/{base_fn}.png', format='png', dpi=1000)
        plt.close(fig)

    ##############
    # utils
    ##############
    @staticmethod
    def expand_f0_ph(f0, mel2ph):
        f0 = denorm_f0(f0, None, hparams)
        f0 = F.pad(f0, [1, 0])
        f0 = torch.gather(f0, 1, mel2ph)  # [B, T_mel]
        return f0


if __name__ == '__main__':
    FastSpeech2Task.start()
