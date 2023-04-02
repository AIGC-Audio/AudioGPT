import matplotlib
matplotlib.use('Agg')
from data_gen.tts.data_gen_utils import get_pitch
from modules.fastspeech.tts_modules import mel2ph_to_dur
import matplotlib.pyplot as plt
from utils import audio
from utils.pitch_utils import norm_interp_f0, denorm_f0, f0_to_coarse
from vocoders.base_vocoder import get_vocoder_cls
import json
from utils.plot import spec_to_figure
from utils.hparams import hparams
import torch
import torch.optim
import torch.nn.functional as F
import torch.utils.data
from modules.GenerSpeech.task.dataset import GenerSpeech_dataset
from modules.GenerSpeech.model.generspeech import GenerSpeech
import torch.distributions
import numpy as np
from utils.tts_utils import select_attn
import utils
import os
from tasks.tts.fs2 import FastSpeech2Task

class GenerSpeechTask(FastSpeech2Task):
    def __init__(self):
        super(GenerSpeechTask, self).__init__()
        self.dataset_cls = GenerSpeech_dataset

    def build_tts_model(self):
        self.model = GenerSpeech(self.phone_encoder)

    def build_model(self):
        self.build_tts_model()
        if hparams['load_ckpt'] != '':
            self.load_ckpt(hparams['load_ckpt'], strict=False)
        utils.num_params(self.model)
        return self.model

    def run_model(self, model, sample, return_output=False):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']  # [B, T_s]
        mel2word = sample['mel2word']
        f0 = sample['f0']  # [B, T_s]
        uv = sample['uv']  # [B, T_s] 0/1

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        emo_embed = sample.get('emo_embed')
        output = model(txt_tokens, mel2ph=mel2ph, ref_mel2ph=mel2ph, ref_mel2word=mel2word, spk_embed=spk_embed, emo_embed=emo_embed,
                       ref_mels=target, f0=f0, uv=uv, tgt_mels=target, global_steps=self.global_step, infer=False)
        losses = {}
        losses['postflow'] = output['postflow']
        if self.global_step > hparams['forcing']:
            losses['gloss'] = (output['gloss_utter'] + output['gloss_ph'] + output['gloss_word']) / 3
        if self.global_step > hparams['vq_start']:
            losses['vq_loss'] = (output['vq_loss_utter'] + output['vq_loss_ph'] + output['vq_loss_word']) / 3
            losses['ppl_utter'] = output['ppl_utter']
            losses['ppl_ph'] = output['ppl_ph']
            losses['ppl_word'] = output['ppl_word']
        self.add_mel_loss(output['mel_out'], target, losses)
        self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
        if hparams['use_pitch_embed']:
            self.add_pitch_loss(output, sample, losses)
        output['select_attn'] = select_attn(output['attn_ph'])

        if not return_output:
            return losses
        else:
            return losses, output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        encdec_attn = model_out['select_attn']
        mel_out = self.model.out2mel(model_out['mel_out'])
        outputs = utils.tensors_to_scalars(outputs)
        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots']:
            vmin = hparams['mel_vmin']
            vmax = hparams['mel_vmax']
            self.plot_mel(batch_idx, sample['mels'], mel_out)
            self.plot_dur(batch_idx, sample, model_out)
            if hparams['use_pitch_embed']:
                self.plot_pitch(batch_idx, sample, model_out)
            if self.vocoder is None:
                self.vocoder = get_vocoder_cls(hparams)()
            if self.global_step > 0:
                spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
                emo_embed = sample.get('emo_embed')
                ref_mels = sample['mels']
                mel2ph = sample['mel2ph']  # [B, T_s]
                mel2word = sample['mel2word']
                # with gt duration
                model_out = self.model(sample['txt_tokens'], mel2ph=mel2ph, ref_mel2ph=mel2ph, ref_mel2word=mel2word, spk_embed=spk_embed,
                                       emo_embed=emo_embed, ref_mels=ref_mels, global_steps=self.global_step, infer=True)
                wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu())
                self.logger.add_audio(f'wav_gtdur_{batch_idx}', wav_pred, self.global_step,
                                      hparams['audio_sample_rate'])
                self.logger.add_figure(f'ali_{batch_idx}', spec_to_figure(encdec_attn[0]), self.global_step)
                self.logger.add_figure(
                    f'mel_gtdur_{batch_idx}',
                    spec_to_figure(model_out['mel_out'][0], vmin, vmax), self.global_step)
                # with pred duration
                model_out = self.model(sample['txt_tokens'], ref_mel2ph=mel2ph, ref_mel2word=mel2word, spk_embed=spk_embed, emo_embed=emo_embed, ref_mels=ref_mels,
                                       global_steps=self.global_step, infer=True)
                self.logger.add_figure(
                    f'mel_{batch_idx}',
                    spec_to_figure(model_out['mel_out'][0], vmin, vmax), self.global_step)
                wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu())
                self.logger.add_audio(f'wav_{batch_idx}', wav_pred, self.global_step, hparams['audio_sample_rate'])
            # gt wav
            if self.global_step <= hparams['valid_infer_interval']:
                mel_gt = sample['mels'][0].cpu()
                wav_gt = self.vocoder.spec2wav(mel_gt)
                self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, 22050)
        return outputs

    ############
    # infer
    ############
    def test_step(self, sample, batch_idx):
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        emo_embed = sample.get('emo_embed')
        txt_tokens = sample['txt_tokens']
        mel2ph, uv, f0 = None, None, None
        ref_mel2word = sample['mel2word']
        ref_mel2ph = sample['mel2ph']
        ref_mels = sample['mels']
        if hparams['use_gt_dur']:
            mel2ph = sample['mel2ph']
        if hparams['use_gt_f0']:
            f0 = sample['f0']
            uv = sample['uv']
        global_steps = 200000
        run_model = lambda: self.model(
            txt_tokens, spk_embed=spk_embed, emo_embed=emo_embed, mel2ph=mel2ph, ref_mel2ph=ref_mel2ph, ref_mel2word=ref_mel2word,
            f0=f0, uv=uv, ref_mels=ref_mels, global_steps=global_steps, infer=True)
        outputs = run_model()
        sample['outputs'] = self.model.out2mel(outputs['mel_out'])
        sample['mel2ph_pred'] = outputs['mel2ph']
        if hparams['use_pitch_embed']:
            sample['f0'] = denorm_f0(sample['f0'], sample['uv'], hparams)
            if hparams['pitch_type'] == 'ph':
                sample['f0'] = torch.gather(F.pad(sample['f0'], [1, 0]), 1, sample['mel2ph'])
            sample['f0_pred'] = outputs.get('f0_denorm')

        return self.after_infer(sample)



    def after_infer(self, predictions, sil_start_frame=0):
        predictions = utils.unpack_dict_to_list(predictions)
        assert len(predictions) == 1, 'Only support batch_size=1 in inference.'
        prediction = predictions[0]
        prediction = utils.tensors_to_np(prediction)
        item_name = prediction.get('item_name')
        text = prediction.get('text')
        ph_tokens = prediction.get('txt_tokens')
        mel_gt = prediction["mels"]
        mel2ph_gt = prediction.get("mel2ph")
        mel2ph_gt = mel2ph_gt if mel2ph_gt is not None else None
        mel_pred = prediction["outputs"]
        mel2ph_pred = prediction.get("mel2ph_pred")
        f0_gt = prediction.get("f0")
        f0_pred = prediction.get("f0_pred")

        str_phs = None
        if self.phone_encoder is not None and 'txt_tokens' in prediction:
            str_phs = self.phone_encoder.decode(prediction['txt_tokens'], strip_padding=True)

        if 'encdec_attn' in prediction:
            encdec_attn = prediction['encdec_attn']  # (1, Tph, Tmel)
            encdec_attn = encdec_attn[encdec_attn.max(-1).sum(-1).argmax(-1)]
            txt_lengths = prediction.get('txt_lengths')
            encdec_attn = encdec_attn.T[:, :txt_lengths]
        else:
            encdec_attn = None

        wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)
        wav_pred[:sil_start_frame * hparams['hop_size']] = 0
        gen_dir = self.gen_dir
        base_fn = f'[{self.results_id:06d}][{item_name}][%s]'
        # if text is not None:
        #     base_fn += text.replace(":", "%3A")[:80]
        base_fn = base_fn.replace(' ', '_')
        if not hparams['profile_infer']:
            os.makedirs(gen_dir, exist_ok=True)
            os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
            os.makedirs(f'{gen_dir}/plot', exist_ok=True)
            if hparams.get('save_mel_npy', False):
                os.makedirs(f'{gen_dir}/npy', exist_ok=True)
            if 'encdec_attn' in prediction:
                os.makedirs(f'{gen_dir}/attn_plot', exist_ok=True)
            self.saving_results_futures.append(
                self.saving_result_pool.apply_async(self.save_result, args=[
                    wav_pred, mel_pred, base_fn % 'TTS', gen_dir, str_phs, mel2ph_pred, encdec_attn]))

            if mel_gt is not None and hparams['save_gt']:
                wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
                self.saving_results_futures.append(
                    self.saving_result_pool.apply_async(self.save_result, args=[
                        wav_gt, mel_gt, base_fn % 'Ref', gen_dir, str_phs, mel2ph_gt]))
                if hparams['save_f0']:
                    import matplotlib.pyplot as plt
                    f0_pred_, _ = get_pitch(wav_pred, mel_pred, hparams)
                    f0_gt_, _ = get_pitch(wav_gt, mel_gt, hparams)
                    fig = plt.figure()
                    plt.plot(f0_pred_, label=r'$\hat{f_0}$')
                    plt.plot(f0_gt_, label=r'$f_0$')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f'{gen_dir}/plot/[F0][{item_name}]{text}.png', format='png')
                    plt.close(fig)

            print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        self.results_id += 1
        return {
            'item_name': item_name,
            'text': text,
            'ph_tokens': self.phone_encoder.decode(ph_tokens.tolist()),
            'wav_fn_pred': base_fn % 'TTS',
            'wav_fn_gt': base_fn % 'Ref',
        }



    @staticmethod
    def save_result(wav_out, mel, base_fn, gen_dir, str_phs=None, mel2ph=None, alignment=None):
        audio.save_wav(wav_out, f'{gen_dir}/wavs/{base_fn}.wav', hparams['audio_sample_rate'],
                       norm=hparams['out_wav_norm'])
        fig = plt.figure(figsize=(14, 10))
        spec_vmin = hparams['mel_vmin']
        spec_vmax = hparams['mel_vmax']
        heatmap = plt.pcolor(mel.T, vmin=spec_vmin, vmax=spec_vmax)
        fig.colorbar(heatmap)
        f0, _ = get_pitch(wav_out, mel, hparams)
        f0 = f0 / 10 * (f0 > 0)
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
        plt.savefig(f'{gen_dir}/plot/{base_fn}.png', format='png')
        plt.close(fig)
        if hparams.get('save_mel_npy', False):
            np.save(f'{gen_dir}/npy/{base_fn}', mel)
        if alignment is not None:
            fig, ax = plt.subplots(figsize=(12, 16))
            im = ax.imshow(alignment, aspect='auto', origin='lower',
                           interpolation='none')
            ax.set_xticks(np.arange(0, alignment.shape[1], 5))
            ax.set_yticks(np.arange(0, alignment.shape[0], 10))
            ax.set_ylabel("$S_p$ index")
            ax.set_xlabel("$H_c$ index")
            fig.colorbar(im, ax=ax)
            fig.savefig(f'{gen_dir}/attn_plot/{base_fn}_attn.png', format='png')
            plt.close(fig)



