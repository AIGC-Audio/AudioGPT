import filecmp

import matplotlib

from utils.plot import spec_to_figure

matplotlib.use('Agg')

from data_gen.tts.data_gen_utils import get_pitch
from modules.fastspeech.tts_modules import mel2ph_to_dur
from tasks.tts.dataset_utils import BaseTTSDataset
from utils.tts_utils import sequence_mask
from multiprocessing.pool import Pool
from tasks.base_task import data_loader, BaseConcatDataset
from utils.common_schedulers import RSQRTSchedule, NoneSchedule
from vocoders.base_vocoder import get_vocoder_cls, BaseVocoder
import os
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from tasks.base_task import BaseTask
from utils.hparams import hparams
from utils.text_encoder import TokenTextEncoder
import json
import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.utils.data
import utils
from utils import audio
import pandas as pd


class TTSBaseTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_cls = BaseTTSDataset
        self.max_tokens = hparams['max_tokens']
        self.max_sentences = hparams['max_sentences']
        self.max_valid_tokens = hparams['max_valid_tokens']
        if self.max_valid_tokens == -1:
            hparams['max_valid_tokens'] = self.max_valid_tokens = self.max_tokens
        self.max_valid_sentences = hparams['max_valid_sentences']
        if self.max_valid_sentences == -1:
            hparams['max_valid_sentences'] = self.max_valid_sentences = self.max_sentences
        self.vocoder = None
        self.phone_encoder = self.build_phone_encoder(hparams['binary_data_dir'])
        self.padding_idx = self.phone_encoder.pad()
        self.eos_idx = self.phone_encoder.eos()
        self.seg_idx = self.phone_encoder.seg()
        self.saving_result_pool = None
        self.saving_results_futures = None
        self.stats = {}

    @data_loader
    def train_dataloader(self):
        if hparams['train_sets'] != '':
            train_sets = hparams['train_sets'].split("|")
            # check if all train_sets have the same spk map and dictionary
            binary_data_dir = hparams['binary_data_dir']
            file_to_cmp = ['phone_set.json']
            if os.path.exists(f'{binary_data_dir}/word_set.json'):
                file_to_cmp.append('word_set.json')
            if hparams['use_spk_id']:
                file_to_cmp.append('spk_map.json')
            for f in file_to_cmp:
                for ds_name in train_sets:
                    base_file = os.path.join(binary_data_dir, f)
                    ds_file = os.path.join(ds_name, f)
                    assert filecmp.cmp(base_file, ds_file), \
                        f'{f} in {ds_name} is not same with that in {binary_data_dir}.'
            train_dataset = BaseConcatDataset([
                self.dataset_cls(prefix='train', shuffle=True, data_dir=ds_name) for ds_name in train_sets])
        else:
            train_dataset = self.dataset_cls(prefix=hparams['train_set_name'], shuffle=True)
        return self.build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences,
                                     endless=hparams['endless_ds'])

    @data_loader
    def val_dataloader(self):
        valid_dataset = self.dataset_cls(prefix=hparams['valid_set_name'], shuffle=False)
        return self.build_dataloader(valid_dataset, False, self.max_valid_tokens, self.max_valid_sentences)

    @data_loader
    def test_dataloader(self):
        test_dataset = self.dataset_cls(prefix=hparams['test_set_name'], shuffle=False)
        self.test_dl = self.build_dataloader(
            test_dataset, False, self.max_valid_tokens,
            self.max_valid_sentences, batch_by_size=False)
        return self.test_dl

    def build_dataloader(self, dataset, shuffle, max_tokens=None, max_sentences=None,
                         required_batch_size_multiple=-1, endless=False, batch_by_size=True):
        devices_cnt = torch.cuda.device_count()
        if devices_cnt == 0:
            devices_cnt = 1
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = devices_cnt

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches

        if max_tokens is not None:
            max_tokens *= devices_cnt
        if max_sentences is not None:
            max_sentences *= devices_cnt
        indices = dataset.ordered_indices()
        if batch_by_size:
            batch_sampler = utils.batch_by_size(
                indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
        else:
            batch_sampler = []
            for i in range(0, len(indices), max_sentences):
                batch_sampler.append(indices[i:i + max_sentences])

        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
            if endless:
                batches = [b for _ in range(1000) for b in shuffle_batches(list(batch_sampler))]
        else:
            batches = batch_sampler
            if endless:
                batches = [b for _ in range(1000) for b in batches]
        num_workers = dataset.num_workers
        if self.trainer.use_ddp:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]
        return torch.utils.data.DataLoader(dataset,
                                           collate_fn=dataset.collater,
                                           batch_sampler=batches,
                                           num_workers=num_workers,
                                           pin_memory=False)

    def build_phone_encoder(self, data_dir):
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        return TokenTextEncoder(None, vocab_list=phone_list, replace_oov=',')

    def build_scheduler(self, optimizer):
        if hparams['scheduler'] == 'rsqrt':
            return RSQRTSchedule(optimizer)
        else:
            return NoneSchedule(optimizer)

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        return optimizer

    def plot_mel(self, batch_idx, spec, spec_out, name=None):
        spec_cat = torch.cat([spec, spec_out], -1)
        name = f'mel_{batch_idx}' if name is None else name
        vmin = hparams['mel_vmin']
        vmax = hparams['mel_vmax']
        self.logger.add_figure(name, spec_to_figure(spec_cat[0], vmin, vmax), self.global_step)

    def test_start(self):
        self.saving_result_pool = Pool(min(int(os.getenv('N_PROC', os.cpu_count())), 16))
        self.saving_results_futures = []
        self.results_id = 0
        self.gen_dir = os.path.join(
            hparams['work_dir'],
            f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()

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
            encdec_attn = prediction['encdec_attn']
            encdec_attn = encdec_attn[encdec_attn.max(-1).sum(-1).argmax(-1)]
            txt_lengths = prediction.get('txt_lengths')
            encdec_attn = encdec_attn.T[:txt_lengths, :len(mel_gt)]
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
                    wav_pred, mel_pred, base_fn % 'P', gen_dir, str_phs, mel2ph_pred, encdec_attn]))

            if mel_gt is not None and hparams['save_gt']:
                wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
                self.saving_results_futures.append(
                    self.saving_result_pool.apply_async(self.save_result, args=[
                        wav_gt, mel_gt, base_fn % 'G', gen_dir, str_phs, mel2ph_gt]))
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
            'wav_fn_pred': base_fn % 'P',
            'wav_fn_gt': base_fn % 'G',
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
            decoded_txt = str_phs.split(" ")
            ax.set_yticks(np.arange(len(decoded_txt)))
            ax.set_yticklabels(list(decoded_txt), fontsize=6)
            fig.colorbar(im, ax=ax)
            fig.savefig(f'{gen_dir}/attn_plot/{base_fn}_attn.png', format='png')
            plt.close(fig)

    def test_end(self, outputs):
        pd.DataFrame(outputs).to_csv(f'{self.gen_dir}/meta.csv')
        self.saving_result_pool.close()
        [f.get() for f in tqdm(self.saving_results_futures)]
        self.saving_result_pool.join()
        return {}

    ##########
    # utils
    ##########
    def weights_nonzero_speech(self, target):
        # target : B x T x mel
        # Assign weight 1.0 to all labels except for padding (id=0).
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

    def make_stop_target(self, target):
        # target : B x T x mel
        seq_mask = target.abs().sum(-1).ne(0).float()
        seq_length = seq_mask.sum(1)
        mask_r = 1 - sequence_mask(seq_length - 1, target.size(1)).float()
        return seq_mask, mask_r
