import torch
from text_to_speech.modules.vocoder.hifigan.hifigan import HifiGanGenerator
from tasks.tts.vocoder_infer.base_vocoder import register_vocoder, BaseVocoder
from text_to_speech.utils.commons.ckpt_utils import load_ckpt
from text_to_speech.utils.commons.hparams import set_hparams, hparams
from text_to_speech.utils.commons.meters import Timer

total_time = 0


@register_vocoder('HifiGAN')
class HifiGAN(BaseVocoder):
    def __init__(self):
        base_dir = hparams['vocoder_ckpt']
        config_path = f'{base_dir}/config.yaml'
        self.config = config = set_hparams(config_path, global_hparams=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HifiGanGenerator(config)
        load_ckpt(self.model, base_dir, 'model_gen')
        self.model.to(self.device)
        self.model.eval()

    def spec2wav(self, mel, **kwargs):
        device = self.device
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).to(device)
            c = c.transpose(2, 1)
            with Timer('hifigan', enable=hparams['profile_infer']):
                y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        return wav_out