import sys
import torch

if __name__ == '__main__':
    ckpt_path = sys.argv[1]
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint = {'state_dict': checkpoint['state_dict']}
    torch.save(checkpoint, ckpt_path, _use_new_zipfile_serialization=False)
