from loss import WeightedCrossEntropy
import random

import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataset import VGGSound
from transforms import Crop, StandardNormalizeAudio, ToTensor
from logger import LoggerWithTBoard
from metrics import metrics
from model import VGGishish

if __name__ == "__main__":
    cfg_cli = OmegaConf.from_cli()
    cfg_yml = OmegaConf.load(cfg_cli.config)
    # the latter arguments are prioritized
    cfg = OmegaConf.merge(cfg_yml, cfg_cli)
    OmegaConf.set_readonly(cfg, True)
    print(OmegaConf.to_yaml(cfg))

    logger = LoggerWithTBoard(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    # makes iterations faster (in this case 30%) if your inputs are of a fixed size
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    torch.backends.cudnn.benchmark = True

    transforms = [
        StandardNormalizeAudio(cfg.mels_path),
    ]
    if cfg.cropped_size not in [None, 'None', 'none']:
        logger.print_logger.info(f'Using cropping {cfg.cropped_size}')
        transforms.append(Crop(cfg.cropped_size))
    transforms.append(ToTensor())
    transforms = torchvision.transforms.transforms.Compose(transforms)

    datasets = {
        'train': VGGSound('train', cfg.mels_path, transforms),
        'valid': VGGSound('valid', cfg.mels_path, transforms),
        'test': VGGSound('test', cfg.mels_path, transforms),
    }

    loaders = {
        'train': DataLoader(datasets['train'], batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                            num_workers=cfg.num_workers, pin_memory=True),
        'valid': DataLoader(datasets['valid'], batch_size=cfg.batch_size,
                            num_workers=cfg.num_workers, pin_memory=True),
        'test': DataLoader(datasets['test'], batch_size=cfg.batch_size,
                           num_workers=cfg.num_workers, pin_memory=True),
    }

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    model = VGGishish(cfg.conv_layers, cfg.use_bn, num_classes=len(datasets['train'].target2label))
    model = model.to(device)
    param_num = logger.log_param_num(model)

    if cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.learning_rate, betas=cfg.betas, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError

    if cfg.cls_weights_in_loss:
        weights = 1 / datasets['train'].class_counts
    else:
        weights = torch.ones(len(datasets['train'].target2label))
    criterion = WeightedCrossEntropy(weights.to(device))

    # loop over the train and validation multiple times (typical PT boilerplate)
    no_change_epochs = 0
    best_valid_loss = float('inf')
    early_stop_triggered = False

    for epoch in range(cfg.num_epochs):

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0
            preds_from_each_batch = []
            targets_from_each_batch = []

            prog_bar = tqdm(loaders[phase], f'{phase} ({epoch})', ncols=0)
            for i, batch in enumerate(prog_bar):
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets, to_weight=phase == 'train')

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # loss
                running_loss += loss.item()

                # for metrics calculation later on
                preds_from_each_batch += [outputs.detach().cpu()]
                targets_from_each_batch += [targets.cpu()]

                # iter logging
                if i % 50 == 0:
                    logger.log_iter_loss(loss.item(), epoch*len(loaders[phase])+i, phase)
                    # tracks loss in the tqdm progress bar
                    prog_bar.set_postfix(loss=loss.item())

            # logging loss
            epoch_loss = running_loss / len(loaders[phase])
            logger.log_epoch_loss(epoch_loss, epoch, phase)

            # logging metrics
            preds_from_each_batch = torch.cat(preds_from_each_batch)
            targets_from_each_batch = torch.cat(targets_from_each_batch)
            metrics_dict = metrics(targets_from_each_batch, preds_from_each_batch)
            logger.log_epoch_metrics(metrics_dict, epoch, phase)

            # Early stopping
            if phase == 'valid':
                if epoch_loss < best_valid_loss:
                    no_change_epochs = 0
                    best_valid_loss = epoch_loss
                    logger.log_best_model(model, epoch_loss, epoch, optimizer, metrics_dict)
                else:
                    no_change_epochs += 1
                    logger.print_logger.info(
                        f'Valid loss hasnt changed for {no_change_epochs} patience: {cfg.patience}'
                    )
                    if no_change_epochs >= cfg.patience:
                        early_stop_triggered = True

        if early_stop_triggered:
            logger.print_logger.info(f'Training is early stopped @ {epoch}')
            break

    logger.print_logger.info('Finished Training')

    # loading the best model
    ckpt = torch.load(logger.best_model_path)
    model.load_state_dict(ckpt['model'])
    logger.print_logger.info(f'Loading the best model from {logger.best_model_path}')
    logger.print_logger.info((f'The model was trained for {ckpt["epoch"]} epochs. Loss: {ckpt["loss"]:.4f}'))

    # Testing the model
    model.eval()
    running_loss = 0
    preds_from_each_batch = []
    targets_from_each_batch = []

    for i, batch in enumerate(loaders['test']):
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, targets, to_weight=False)

        # loss
        running_loss += loss.item()

        # for metrics calculation later on
        preds_from_each_batch += [outputs.detach().cpu()]
        targets_from_each_batch += [targets.cpu()]

    # logging metrics
    preds_from_each_batch = torch.cat(preds_from_each_batch)
    targets_from_each_batch = torch.cat(targets_from_each_batch)
    test_metrics_dict = metrics(targets_from_each_batch, preds_from_each_batch)
    test_metrics_dict['avg_loss'] = running_loss / len(loaders['test'])
    test_metrics_dict['param_num'] = param_num
    # TODO: I have no idea why tboard doesn't keep metrics (hparams) when
    # I run this experiment from cli: `python train_vggishish.py config=./configs/vggish.yaml`
    # while when I run it in vscode debugger the metrics are logger (wtf)
    logger.log_test_metrics(test_metrics_dict, dict(cfg), ckpt['epoch'])

    logger.print_logger.info('Finished the experiment')
