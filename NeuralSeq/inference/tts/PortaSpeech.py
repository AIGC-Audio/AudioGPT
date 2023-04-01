import torch
from inference.tts.base_tts_infer import BaseTTSInfer
from utils.ckpt_utils import load_ckpt
from modules.portaspeech.portaspeech import PortaSpeech

class TTSInference(BaseTTSInfer):
    def __init__(self, hparams, device=None):
        super().__init__(hparams, device)
        print("Initializing TTS model to %s" % device)
        self.spk_map = self.preprocessor.load_spk_map(self.data_dir)
        print("TTS loaded!")

    def build_model(self):
        model = PortaSpeech(self.ph_encoder, self.word_encoder)
        load_ckpt(model, self.hparams['work_dir'], 'model')
        with torch.no_grad():
            model.store_inverse_all()
        return model
    
    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        with torch.no_grad():
            output = self.model(
                sample['txt_tokens'],
                sample['word_tokens'],
                ph2word=sample['ph2word'],
                word_len=sample['word_lengths'].max(),
                infer=True,
                forward_post_glow=True,
                spk_id=sample.get('spk_ids')
            )
            mel_out = output['mel_out']
            wav_out = self.run_vocoder(mel_out)
        wav_out = wav_out.cpu().numpy()
        return wav_out[0]

    def preprocess_input(self, inp):
        """

        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """
        preprocessor, preprocess_args = self.preprocessor, self.preprocess_args
        text_raw = inp['text']
        item_name = inp.get('item_name', '<ITEM_NAME>')
        spk_name = inp.get('spk_name', '<SINGLE_SPK>')
        ph, txt, word, ph2word, ph_gb_word = preprocessor.txt_to_ph(
            preprocessor.txt_processor, text_raw, preprocess_args)
        word_token = self.word_encoder.encode(word)
        ph_token = self.ph_encoder.encode(ph)
        spk_id = self.spk_map[spk_name]
        item = {'item_name': item_name, 'text': txt, 'ph': ph, 'spk_id': spk_id,
                'ph_token': ph_token, 'word_token': word_token, 'ph2word': ph2word,
                'ph_words':ph_gb_word, 'words': word}
        item['ph_len'] = len(item['ph_token'])
        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        word_tokens = torch.LongTensor(item['word_token'])[None, :].to(self.device)
        word_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        ph2word = torch.LongTensor(item['ph2word'])[None, :].to(self.device)
        spk_ids = torch.LongTensor(item['spk_id'])[None, :].to(self.device)
        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'word_tokens': word_tokens,
            'word_lengths': word_lengths,
            'ph2word': ph2word,
            'spk_ids': spk_ids,
        }
        return batch

    def postprocess_output(self, output):
        return output



