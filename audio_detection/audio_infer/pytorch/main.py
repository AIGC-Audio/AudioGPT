import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
 
from utilities import (create_folder, get_filename, create_logging, Mixup, 
    StatisticsContainer)
from models import (PVT, PVT2, PVT_lr, PVT_nopretrain, PVT_2layer, Cnn14, Cnn14_no_specaug, Cnn14_no_dropout, 
    Cnn6, Cnn10, ResNet22, ResNet38, ResNet54, Cnn14_emb512, Cnn14_emb128, 
    Cnn14_emb32, MobileNetV1, MobileNetV2, LeeNet11, LeeNet24, DaiNet19, 
    Res1dNet31, Res1dNet51, Wavegram_Cnn14, Wavegram_Logmel_Cnn14, 
    Wavegram_Logmel128_Cnn14, Cnn14_16k, Cnn14_8k, Cnn14_mel32, Cnn14_mel128, 
    Cnn14_mixup_time_domain, Cnn14_DecisionLevelMax, Cnn14_DecisionLevelAtt, Cnn6_Transformer, GLAM, GLAM2, GLAM3, Cnn4, EAT)
#from models_test import (PVT_test)
#from models1 import (PVT1)
#from models_vig import (VIG, VIG2)
#from models_vvt import (VVT)
#from models2 import (MPVIT, MPVIT2)
#from models_reshape import (PVT_reshape, PVT_tscam)
#from models_swin import (Swin, Swin_nopretrain)
#from models_swin2 import (Swin2)
#from models_van import (Van, Van_tiny)
#from models_focal import (Focal)
#from models_cross import (Cross)
#from models_cov import (Cov)
#from models_cnn import (Cnn_light)
#from models_twins import (Twins)
#from models_cmt import (Cmt, Cmt1)
#from models_shunted import (Shunted)
#from models_quadtree import (Quadtree, Quadtree2, Quadtree_nopretrain)
#from models_davit import (Davit_tscam, Davit, Davit_nopretrain)
from pytorch_utils import (move_data_to_device, count_parameters, count_flops, 
    do_mixup)
from data_generator import (AudioSetDataset, TrainSampler, BalancedTrainSampler, 
    AlternateTrainSampler, EvaluateSampler, collate_fn)
from evaluate import Evaluator
import config
from losses import get_loss_func


def train(args):
    """Train AudioSet tagging model. 

    Args:
      dataset_dir: str
      workspace: str
      data_type: 'balanced_train' | 'full_train'
      window_size: int
      hop_size: int
      mel_bins: int
      model_type: str
      loss_type: 'clip_bce'
      balanced: 'none' | 'balanced' | 'alternate'
      augmentation: 'none' | 'mixup'
      batch_size: int
      learning_rate: float
      resume_iteration: int
      early_stop: int
      accumulation_steps: int
      cuda: bool
    """

    # Arugments & parameters
    workspace = args.workspace
    data_type = args.data_type
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename

    num_workers = 8
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    loss_func = get_loss_func(loss_type)

    # Paths
    black_list_csv = None
    
    train_indexes_hdf5_path = os.path.join(workspace, 'hdf5s', 'indexes', 
        '{}.h5'.format(data_type))

    eval_bal_indexes_hdf5_path = os.path.join(workspace, 
        'hdf5s', 'indexes', 'balanced_train.h5')

    eval_test_indexes_hdf5_path = os.path.join(workspace, 'hdf5s', 'indexes', 
        'eval.h5')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)
    
    statistics_path = os.path.join(workspace, 'statistics', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))

    create_logging(logs_dir, filemode='w')
    logging.info(args)
    
    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
        device = 'cpu'
    
    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    logging.info("Total params: %.2fM" % (total/1e6))
    #params_num = count_parameters(model)
    # flops_num = count_flops(model, clip_samples)
    #logging.info('Parameters num: {}'.format(params_num))
    # logging.info('Flops num: {:.3f} G'.format(flops_num / 1e9))
    
    # Dataset will be used by DataLoader later. Dataset takes a meta as input 
    # and return a waveform and a target.
    dataset = AudioSetDataset(sample_rate=sample_rate)

    # Train sampler
    if balanced == 'none':
        Sampler = TrainSampler
    elif balanced == 'balanced':
        Sampler = BalancedTrainSampler
    elif balanced == 'alternate':
        Sampler = AlternateTrainSampler
     
    train_sampler = Sampler(
        indexes_hdf5_path=train_indexes_hdf5_path, 
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size,
        black_list_csv=black_list_csv)
    
    # Evaluate sampler
    eval_bal_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_bal_indexes_hdf5_path, batch_size=batch_size)

    eval_test_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_test_indexes_hdf5_path, batch_size=batch_size)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)
    
    eval_bal_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=eval_bal_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    eval_test_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=eval_test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)
    mix=0.5
    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=mix)
    print(mix)
    logging.info(mix)

    # Evaluator
    evaluator = Evaluator(model=model)
        
    # Statistics
    statistics_container = StatisticsContainer(statistics_path)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-06, verbose=True)
    train_bgn_time = time.time()
    
    # Resume training
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            '{}_iterations.pth'.format(resume_iteration))

        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = checkpoint['iteration']

    else:
        iteration = 0
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)

    if resume_iteration:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(optimizer.state_dict()['param_groups'][0]['lr'])

    time1 = time.time()
    
    for batch_data_dict in train_loader:
        """batch_data_dict: {
            'audio_name': (batch_size [*2 if mixup],), 
            'waveform': (batch_size [*2 if mixup], clip_samples), 
            'target': (batch_size [*2 if mixup], classes_num), 
            (ifexist) 'mixup_lambda': (batch_size * 2,)}
        """
        
        # Evaluate
        if (iteration % 2000 == 0 and iteration >= resume_iteration) or (iteration == 0):
            train_fin_time = time.time()

            bal_statistics = evaluator.evaluate(eval_bal_loader)
            test_statistics = evaluator.evaluate(eval_test_loader)
                            
            logging.info('Validate bal mAP: {:.3f}'.format(
                np.mean(bal_statistics['average_precision'])))

            logging.info('Validate test mAP: {:.3f}'.format(
                np.mean(test_statistics['average_precision'])))

            statistics_container.append(iteration, bal_statistics, data_type='bal')
            statistics_container.append(iteration, test_statistics, data_type='test')
            statistics_container.dump()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()
        
        # Save model
        if iteration % 2000 == 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict(), 
                'sampler': train_sampler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
        
        # Mixup lambda
        if 'mixup' in augmentation:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                batch_size=len(batch_data_dict['waveform']))

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
        
        # Forward
        model.train()

        if 'mixup' in augmentation:
            batch_output_dict = model(batch_data_dict['waveform'], 
                batch_data_dict['mixup_lambda'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': do_mixup(batch_data_dict['target'], 
                batch_data_dict['mixup_lambda'])}
            """{'target': (batch_size, classes_num)}"""
        else:
            batch_output_dict = model(batch_data_dict['waveform'], None)
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': batch_data_dict['target']}
            """{'target': (batch_size, classes_num)}"""

        # Loss
        loss = loss_func(batch_output_dict, batch_target_dict)
        # Backward
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        if iteration % 10 == 0:
            print(iteration, loss)
            #print('--- Iteration: {}, train time: {:.3f} s / 10 iterations ---'\
            #    .format(iteration, time.time() - time1))
            #time1 = time.time()

        if iteration % 2000 == 0:
            scheduler.step(np.mean(test_statistics['average_precision']))
            print(optimizer.state_dict()['param_groups'][0]['lr'])
            logging.info(optimizer.state_dict()['param_groups'][0]['lr'])        

        # Stop learning
        if iteration == early_stop:
            break

        iteration += 1
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--data_type', type=str, default='full_train', choices=['balanced_train', 'full_train'])
    parser_train.add_argument('--sample_rate', type=int, default=32000)
    parser_train.add_argument('--window_size', type=int, default=1024)
    parser_train.add_argument('--hop_size', type=int, default=320)
    parser_train.add_argument('--mel_bins', type=int, default=64)
    parser_train.add_argument('--fmin', type=int, default=50)
    parser_train.add_argument('--fmax', type=int, default=14000) 
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, default='clip_bce', choices=['clip_bce'])
    parser_train.add_argument('--balanced', type=str, default='balanced', choices=['none', 'balanced', 'alternate'])
    parser_train.add_argument('--augmentation', type=str, default='mixup', choices=['none', 'mixup'])
    parser_train.add_argument('--batch_size', type=int, default=32)
    parser_train.add_argument('--learning_rate', type=float, default=1e-3)
    parser_train.add_argument('--resume_iteration', type=int, default=0)
    parser_train.add_argument('--early_stop', type=int, default=1000000)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')