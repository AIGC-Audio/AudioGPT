"""This script is the data preparation script for Deep3DFaceRecon_pytorch
"""

import os 
import numpy as np
import argparse
from util.detect_lm68 import detect_68p,load_lm_graph
from util.skin_mask import get_skin_mask
from util.generate_list import check_list, write_list
import warnings
warnings.filterwarnings("ignore") 

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='datasets', help='root directory for training data')
parser.add_argument('--img_folder', nargs="+", required=True, help='folders of training images')
parser.add_argument('--mode', type=str, default='train', help='train or val')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def data_prepare(folder_list,mode):

    lm_sess,input_op,output_op = load_lm_graph('./checkpoints/lm_model/68lm_detector.pb') # load a tensorflow version 68-landmark detector

    for img_folder in folder_list:
        detect_68p(img_folder,lm_sess,input_op,output_op) # detect landmarks for images
        get_skin_mask(img_folder) # generate skin attention mask for images

    # create files that record path to all training data
    msks_list = []
    for img_folder in folder_list:
        path = os.path.join(img_folder, 'mask')
        msks_list += ['/'.join([img_folder, 'mask', i]) for i in sorted(os.listdir(path)) if 'jpg' in i or 
                                                    'png' in i or 'jpeg' in i or 'PNG' in i]

    imgs_list = [i.replace('mask/', '') for i in msks_list]
    lms_list = [i.replace('mask', 'landmarks') for i in msks_list]
    lms_list = ['.'.join(i.split('.')[:-1]) + '.txt' for i in lms_list]
    
    lms_list_final, imgs_list_final, msks_list_final = check_list(lms_list, imgs_list, msks_list) # check if the path is valid
    write_list(lms_list_final, imgs_list_final, msks_list_final, mode=mode) # save files

if __name__ == '__main__':
    print('Datasets:',opt.img_folder)
    data_prepare([os.path.join(opt.data_root,folder) for folder in opt.img_folder],opt.mode)
