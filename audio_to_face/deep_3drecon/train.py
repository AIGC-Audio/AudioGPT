"""This script is the training script for Deep3DFaceRecon_pytorch
"""

import os
import time
import numpy as np
import torch
from options.train_options import TrainOptions
from data import create_dataset
from deep_3drecon_models import create_model
from util.visualizer import MyVisualizer
from util.util import genvalconf
import torch.multiprocessing as mp
import torch.distributed as dist


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, train_opt):
    val_opt = genvalconf(train_opt, isTrain=False)
    
    device = torch.device(rank)
    torch.cuda.set_device(device)
    use_ddp = train_opt.use_ddp
    
    if use_ddp:
        setup(rank, world_size, train_opt.ddp_port)

    train_dataset, val_dataset = create_dataset(train_opt, rank=rank), create_dataset(val_opt, rank=rank)
    train_dataset_batches, val_dataset_batches = \
        len(train_dataset) // train_opt.batch_size, len(val_dataset) // val_opt.batch_size
    
    model = create_model(train_opt)   # create a model given train_opt.model and other options
    model.setup(train_opt)
    model.device = device
    model.parallelize()

    if rank == 0:
        print('The batch number of training images = %d\n, \
            the batch number of validation images = %d'% (train_dataset_batches, val_dataset_batches))
        model.print_networks(train_opt.verbose)
        visualizer = MyVisualizer(train_opt)   # create a visualizer that display/save images and plots

    total_iters = train_dataset_batches * (train_opt.epoch_count - 1)   # the total number of training iterations
    t_data = 0
    t_val = 0
    optimize_time = 0.1
    batch_size = 1 if train_opt.display_per_batch else train_opt.batch_size

    if use_ddp:
        dist.barrier()

    times = []
    for epoch in range(train_opt.epoch_count, train_opt.n_epochs + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for train_data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        train_dataset.set_epoch(epoch)
        for i, train_data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % train_opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += batch_size
            epoch_iter += batch_size

            torch.cuda.synchronize()
            optimize_start_time = time.time()

            model.set_input(train_data)  # unpack train_data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if use_ddp:
                dist.barrier()

            if rank == 0 and (total_iters == batch_size or total_iters % train_opt.display_freq == 0):   # display images on visdom and save images to a HTML file
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), total_iters, epoch,
                    save_results=True,
                    add_image=train_opt.add_image)
                    # (total_iters == batch_size or total_iters % train_opt.evaluation_freq == 0)
            
            if rank == 0 and (total_iters == batch_size or total_iters % train_opt.print_freq == 0):    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters == batch_size or total_iters % train_opt.evaluation_freq == 0:
                with torch.no_grad():
                    torch.cuda.synchronize()
                    val_start_time = time.time()
                    losses_avg = {}
                    model.eval()
                    for j, val_data in enumerate(val_dataset):
                        model.set_input(val_data)
                        model.optimize_parameters(isTrain=False)
                        if rank == 0 and j < train_opt.vis_batch_nums:
                            model.compute_visuals()
                            visualizer.display_current_results(model.get_current_visuals(), total_iters, epoch,
                                    dataset='val', save_results=True, count=j * val_opt.batch_size,
                                    add_image=train_opt.add_image)

                        if j < train_opt.eval_batch_nums:
                            losses = model.get_current_losses()
                            for key, value in losses.items():
                                losses_avg[key] = losses_avg.get(key, 0) + value

                    for key, value in losses_avg.items():
                        losses_avg[key] = value / min(train_opt.eval_batch_nums, val_dataset_batches)

                    torch.cuda.synchronize()
                    eval_time = time.time() - val_start_time
                    
                    if rank == 0:
                        visualizer.print_current_losses(epoch, epoch_iter, losses_avg, eval_time, t_data, dataset='val') # visualize training results
                        visualizer.plot_current_losses(total_iters, losses_avg, dataset='val')
                model.train()      

            if use_ddp:
                dist.barrier()

            if rank == 0 and (total_iters == batch_size or total_iters % train_opt.save_latest_freq == 0):   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(train_opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if train_opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            
            if use_ddp:
                dist.barrier()
            
            iter_data_time = time.time()

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, train_opt.n_epochs, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
        
        if rank == 0 and epoch % train_opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        if use_ddp:
            dist.barrier()

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")
    
    train_opt = TrainOptions().parse()   # get training options
    world_size = train_opt.world_size               

    if train_opt.use_ddp:
        mp.spawn(main, args=(world_size, train_opt), nprocs=world_size, join=True)
    else:
        main(0, world_size, train_opt)
