conda create -n geneface python=3.9 -y
conda activate geneface
# install pytorch with conda, older versions also work
conda install pytorch=1.12 torchvision cudatoolkit=11.3.1 -c pytorch -c nvidia -y
# install pytorch-3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y # 0.7.2 recommended
# other dependencies, including tensorflow-gpu=2.x
sudo apt-get install libasound2-dev portaudio19-dev # dependency for pyaudio
pip install -r requirements.txt 
bash docs/prepare_env/install_ext.sh


scp yezhenhui@192.168.1.163:/home/yezhenhui/projects/GeneFace/deep_3drecon/BFM/Exp_Pca.bin audio_to_face/deep_3drecon/BFM/Exp_Pca.bin
scp yezhenhui@192.168.1.163:/home/yezhenhui/projects/GeneFace/deep_3drecon/BFM/01_MorphableModel.mat audio_to_face/deep_3drecon/BFM/01_MorphableModel.mat
scp yezhenhui@192.168.1.163:/home/yezhenhui/projects/GeneFace/deep_3drecon/BFM/BFM_model_front.mat audio_to_face/deep_3drecon/BFM/BFM_model_front.mat
scp yezhenhui@192.168.1.163:/home/yezhenhui/projects/GeneFace/deep_3drecon/network/FaceReconModel.pb audio_to_face/deep_3drecon/network/FaceReconModel.pb

scp yezhenhui@192.168.1.163:/home/yezhenhui/projects/GeneFace/data/binary/videos/May/trainval_dataset.npy audio_to_face/data/binary/videos/May/trainval_dataset.npy

