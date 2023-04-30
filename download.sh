mkdir checkpoints
mkdir audio
mkdir image
mkdir text_to_audio
#  Text to sing
wget -P checkpoints/0831_opencpop_ds1000/ -i https://huggingface.co/spaces/Silentlin/DiffSinger/resolve/main/checkpoints/0831_opencpop_ds1000/config.yaml https://huggingface.co/spaces/Silentlin/DiffSinger/resolve/main/checkpoints/0831_opencpop_ds1000/model_ckpt_steps_320000.ckpt
wget -P checkpoints/0109_hifigan_bigpopcs_hop128/ -i https://huggingface.co/spaces/Silentlin/DiffSinger/blob/main/checkpoints/0109_hifigan_bigpopcs_hop128/config.yaml https://huggingface.co/spaces/Silentlin/DiffSinger/resolve/main/checkpoints/0109_hifigan_bigpopcs_hop128/model_ckpt_steps_1512000.ckpt
wget -P checkpoints/0102_xiaoma_pe/ -i https://huggingface.co/spaces/Silentlin/DiffSinger/blob/main/checkpoints/0102_xiaoma_pe/config.yaml https://huggingface.co/spaces/Silentlin/DiffSinger/resolve/main/checkpoints/0102_xiaoma_pe/model_ckpt_steps_60000.ckpt
#  Text to audio
cd text_to_audio
wget -P text_to_audio/Make_An_Audio/useful_ckpts/ -i https://huggingface.co/spaces/DiffusionSpeech/Make_An_Audio/resolve/main/useful_ckpts/ta40multi_epoch=000085.ckpt
wget -P text_to_audio/Make_An_Audio/useful_ckpts/CLAP/ -i https://huggingface.co/spaces/DiffusionSpeech/Make_An_Audio/resolve/main/useful_ckpts/CLAP/CLAP_weights_2022.pth
wget -P text_to_audio/Make_An_Audio/useful_ckpts/ -i https://huggingface.co/spaces/DiffusionSpeech/Make_An_Audio_img/resolve/main/useful_ckpts/ta54_epoch=000216.ckpt
wget -P text_to_audio/Make_An_Audio/useful_ckpts/ -i https://huggingface.co/spaces/DiffusionSpeech/Make_An_Audio_inpaint/resolve/main/useful_ckpts/inpaint7_epoch00047.ckpt
#  Text to speech
wget -P checkpoints/GenerSpeech/ -i https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/blob/main/checkpoints/GenerSpeech/config.yaml https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/checkpoints/GenerSpeech/model_ckpt_steps_300000.ckpt
wget -P checkpoints/trainset_hifigan/ -i https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/blob/main/checkpoints/trainset_hifigan/config.yaml https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/checkpoints/trainset_hifigan/model_ckpt_steps_1000000.ckpt
wget -P checkpoints/ https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/checkpoints/Emotion_encoder.pt
wget -P data/binary/training_set https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/data/binary/training_set/mfa_dict.txt
wget -P data/binary/training_set https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/data/binary/training_set/mfa_model.zip
wget -P data/binary/training_set https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/data/binary/training_set/phone_set.json
wget -P data/binary/training_set https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/data/binary/training_set/spk_map.json
wget -P data/binary/training_set https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/data/binary/training_set/train_f0s_mean_std.npy
wget -P data/binary/training_set https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/data/binary/training_set/word_set.json
wget -P text_to_speech/checkpoints/hifi_lj -i https://huggingface.co/AIGC-Audio/AudioGPT/blob/main/text_to_speech/checkpoints/hifi_lj/config.yaml https://huggingface.co/AIGC-Audio/AudioGPT/resolve/main/text_to_speech/checkpoints/hifi_lj/model_ckpt_steps_2076000.ckpt
wget -P text_to_speech/checkpoints/ljspeech/ps_adv_baseline -i https://huggingface.co/AIGC-Audio/AudioGPT/blob/main/text_to_speech/checkpoints/ljspeech/ps_adv_baseline/config.yaml https://huggingface.co/AIGC-Audio/AudioGPT/resolve/main/checkpoints/ljspeech/ps_adv_baseline/model_ckpt_steps_160000.ckpt https://huggingface.co/AIGC-Audio/AudioGPT/resolve/main/checkpoints/ljspeech/ps_adv_baseline/model_ckpt_steps_160001.ckpt
# Audio to text
wget -P audio_to_text/audiocaps_cntrstv_cnn14rnn_trm -i https://huggingface.co/AIGC-Audio/AudioGPT/blob/main/audio_to_text/audiocaps_cntrstv_cnn14rnn_trm/config.yaml https://huggingface.co/AIGC-Audio/AudioGPT/resolve/main/audio_to_text/audiocaps_cntrstv_cnn14rnn_trm/swa.pth
wget -P audio_to_text/clotho_cntrstv_cnn14rnn_trm -i https://huggingface.co/AIGC-Audio/AudioGPT/blob/main/audio_to_text/clotho_cntrstv_cnn14rnn_trm/config.yaml https://huggingface.co/AIGC-Audio/AudioGPT/resolve/main/audio_to_text/clotho_cntrstv_cnn14rnn_trm/swa.pth
wget -P audio_to_text/pretrained_feature_extractors https://huggingface.co/AIGC-Audio/AudioGPT/resolve/main/audio_to_text/pretrained_feature_extractors/contrastive_pretrain_cnn14_bertm.pth
# Audio detection
cd audio_detection/audio_infer/useful_ckpts
wget https://huggingface.co/Dongchao/pre_trained_model/resolve/main/audio_detection.pth
cd mono2binaural/useful_ckpts
wget https://huggingface.co/Dongchao/pre_trained_model/resolve/main/m2b.tar.gz
tar -zxvf m2b.tar.gz ./
rm m2b.tar.gz
cd audio_detection/target_sound_detection/useful_ckpts
wget https://huggingface.co/Dongchao/pre_trained_model/resolve/main/tsd.tar.gz
tar -zxvf tsd.tar.gz ./
rm tsd.tar.gz
cd sound_extraction/useful_ckpts
wget https://huggingface.co/Dongchao/pre_trained_model/resolve/main/LASSNet.pt