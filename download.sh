mkdir checkpoints
mkdir audio
mkdir image
mkdir text_to_audio
wget -P checkpoints/0831_opencpop_ds1000/ -i https://huggingface.co/spaces/Silentlin/DiffSinger/resolve/main/checkpoints/0831_opencpop_ds1000/config.yaml https://huggingface.co/spaces/Silentlin/DiffSinger/resolve/main/checkpoints/0831_opencpop_ds1000/model_ckpt_steps_320000.ckpt
wget -P checkpoints/0109_hifigan_bigpopcs_hop128/ -i https://huggingface.co/spaces/Silentlin/DiffSinger/blob/main/checkpoints/0109_hifigan_bigpopcs_hop128/config.yaml https://huggingface.co/spaces/Silentlin/DiffSinger/resolve/main/checkpoints/0109_hifigan_bigpopcs_hop128/model_ckpt_steps_1512000.ckpt
wget -P checkpoints/0102_xiaoma_pe/ -i https://huggingface.co/spaces/Silentlin/DiffSinger/blob/main/checkpoints/0102_xiaoma_pe/config.yaml https://huggingface.co/spaces/Silentlin/DiffSinger/resolve/main/checkpoints/0102_xiaoma_pe/model_ckpt_steps_60000.ckpt
cd text_to_audio
git clone https://huggingface.co/spaces/DiffusionSpeech/Make_An_Audio
git clone https://huggingface.co/spaces/DiffusionSpeech/Make_An_Audio_img
git clone https://huggingface.co/spaces/DiffusionSpeech/Make_An_Audio_inpaint
wget -P text_to_audio/Make_An_Audio/useful_ckpts/ -i https://huggingface.co/spaces/DiffusionSpeech/Make_An_Audio/resolve/main/useful_ckpts/ta40multi_epoch=000085.ckpt
wget -P text_to_audio/Make_An_Audio/useful_ckpts/CLAP/ -i https://huggingface.co/spaces/DiffusionSpeech/Make_An_Audio/resolve/main/useful_ckpts/CLAP/CLAP_weights_2022.pth
wget -P text_to_audio/Make_An_Audio_img/useful_ckpts/ -i https://huggingface.co/spaces/DiffusionSpeech/Make_An_Audio_img/resolve/main/useful_ckpts/ta54_epoch=000216.ckpt
wget -P text_to_audio/Make_An_Audio_img/useful_ckpts/CLAP/ -i https://huggingface.co/spaces/DiffusionSpeech/Make_An_Audio_img/blob/main/useful_ckpts/CLAP/CLAP_weights_2022.pth
wget -P text_to_audio/Make_An_Audio_inpaint/useful_ckpts/ -i https://huggingface.co/spaces/DiffusionSpeech/Make_An_Audio_inpaint/resolve/main/useful_ckpts/inpaint7_epoch00047.ckpt

wget -P checkpoints/GenerSpeech/ -i https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/blob/main/checkpoints/GenerSpeech/config.yaml https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/checkpoints/GenerSpeech/model_ckpt_steps_300000.ckpt
wget -P checkpoints/trainset_hifigan/ -i https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/blob/main/checkpoints/trainset_hifigan/config.yaml https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/checkpoints/trainset_hifigan/model_ckpt_steps_1000000.ckpt
wget -P checkpoints/ https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/checkpoints/Emotion_encoder.pt
wget -P data/binary/training_set https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/data/binary/training_set/mfa_dict.txt
wget -P data/binary/training_set https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/data/binary/training_set/mfa_model.zip
wget -P data/binary/training_set https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/data/binary/training_set/phone_set.json
wget -P data/binary/training_set https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/data/binary/training_set/spk_map.json
wget -P data/binary/training_set https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/data/binary/training_set/train_f0s_mean_std.npy
wget -P data/binary/training_set https://huggingface.co/spaces/Rongjiehuang/GenerSpeech/resolve/main/data/binary/training_set/word_set.json

