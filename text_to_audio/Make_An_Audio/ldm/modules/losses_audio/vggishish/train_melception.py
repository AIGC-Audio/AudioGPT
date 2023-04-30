import random

import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from torch.utils.data.dataloader import DataLoader
from torchvision.models.inception import BasicConv2d, Inception3
from tqdm import tqdm

from dataset import VGGSound
from logger import LoggerWithTBoard
from loss import WeightedCrossEntropy
from metrics import metrics
from transforms import Crop, StandardNormalizeAudio, ToTensor


# TODO: refactor  ./evaluation/feature_extractors/melception.py to handle this class as well.
# So far couldn't do it because of the difference in outputs
class Melception(Inception3):

    def __init__(self, num_classes, **kwargs):
        # inception = Melception(num_classes=309)
        super().__init__(num_classes=num_classes, **kwargs)
        # the same as https://github.com/pytorch/vision/blob/5339e63148/torchvision/models/inception.py#L95
        # but for 1-channel input instead of RGB.
        self.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)
        # also the 'hight' of the mel spec is 80 (vs 299 in RGB) we remove all max pool from Inception
        self.maxpool1 = torch.nn.Identity()
        self.maxpool2 = torch.nn.Identity()

    def forward(self, x):
        x = x.unsqueeze(1)
        return super().forward(x)

def train_inception_scorer(cfg):
    logger = LoggerWithTBoard(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    # makes iterations faster (in this case 30%) if your inputs are of a fixed size
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    torch.backends.cudnn.benchmark = True

    meta_path = './data/vggsound.csv'
    train_ids_path = './data/vggsound_train.txt'
    cache_path = './data/'
    splits_path = cache_path

    transforms = [
        StandardNormalizeAudio(cfg.mels_path, train_ids_path, cache_path),
    ]
    if cfg.cropped_size not in [None, 'None', 'none']:
        logger.print_logger.info(f'Using cropping {cfg.cropped_size}')
        transforms.append(Crop(cfg.cropped_size))
    transforms.append(ToTensor())
    transforms = torchvision.transforms.transforms.Compose(transforms)

    datasets = {
        'train': VGGSound('train', cfg.mels_path, transforms, splits_path, meta_path),
        'valid': VGGSound('valid', cfg.mels_path, transforms, splits_path, meta_path),
        'test': VGGSound('test', cfg.mels_path, transforms, splits_path, meta_path),
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

    model = Melception(num_classes=len(datasets['train'].target2label))
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
                    # inception v3
                    if phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, targets)
                        loss2 = criterion(aux_outputs, targets)
                        loss = loss1 + 0.4*loss2
                        loss = criterion(outputs, targets, to_weight=True)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets, to_weight=False)

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
    # I run this experiment from cli: `python train_melception.py config=./configs/vggish.yaml`
    # while when I run it in vscode debugger the metrics are logger (wtf)
    logger.log_test_metrics(test_metrics_dict, dict(cfg), ckpt['epoch'])

    logger.print_logger.info('Finished the experiment')


if __name__ == '__main__':
    # input = torch.rand(16, 1, 80, 848)
    # output, aux = inception(input)
    # print(output.shape, aux.shape)
    # Expected input size: (3, 299, 299) in RGB -> (1, 80, 848) in Mel Spec
    # train_inception_scorer()

    cfg_cli = OmegaConf.from_cli()
    cfg_yml = OmegaConf.load(cfg_cli.config)
    # the latter arguments are prioritized
    cfg = OmegaConf.merge(cfg_yml, cfg_cli)
    OmegaConf.set_readonly(cfg, True)
    print(OmegaConf.to_yaml(cfg))

    train_inception_scorer(cfg)
