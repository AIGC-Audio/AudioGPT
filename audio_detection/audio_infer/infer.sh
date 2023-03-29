CUDA_VISIBLE_DEVICES=0 python3 pytorch/inference.py sound_event_detection \
    --model_type=PVT \
    --checkpoint_path=/apdcephfs_cq2/share_1297902/speech_user/shaunxliu/dongchao/audio_chatgpt/ft_local/audio_infer/220000_iterations.pth \
    --audio_path="/apdcephfs_cq2/share_1297902/speech_user/shaunxliu/dongchao/audio_chatgpt/ft_local/audio_infer/YDlWd7Wmdi1E.wav" \
    --cuda
