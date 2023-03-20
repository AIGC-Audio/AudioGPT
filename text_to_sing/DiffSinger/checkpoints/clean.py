import sys
import torch

if __name__ == '__main__':
    ckpt_path = sys.argv[1]
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print(checkpoint['state_dict'].keys())
    if 'model' in checkpoint['state_dict']:
        checkpoint = {'state_dict': {'model': checkpoint['state_dict']['model']}}
    else:
        checkpoint = {'state_dict': {'model_gen': checkpoint['state_dict']['model_gen']}}
    torch.save(checkpoint, ckpt_path, _use_new_zipfile_serialization=False)
