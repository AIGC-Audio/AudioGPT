import logging
import os
import time
from shutil import copytree, ignore_patterns

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter, summary


class LoggerWithTBoard(SummaryWriter):

    def __init__(self, cfg):
        # current time stamp and experiment log directory
        self.start_time = time.strftime('%y-%m-%dT%H-%M-%S', time.localtime())
        self.logdir = os.path.join(cfg.logdir, self.start_time)
        # init tboard
        super().__init__(self.logdir)
        # backup the cfg
        OmegaConf.save(cfg, os.path.join(self.log_dir, 'cfg.yaml'))
        # backup the code state
        if cfg.log_code_state:
            dest_dir = os.path.join(self.logdir, 'code')
            copytree(os.getcwd(), dest_dir, ignore=ignore_patterns(*cfg.patterns_to_ignore))

        # init logger which handles printing and logging mostly same things to the log file
        self.print_logger = logging.getLogger('main')
        self.print_logger.setLevel(logging.INFO)
        msgfmt = '[%(levelname)s] %(asctime)s - %(name)s \n    %(message)s'
        datefmt = '%d %b %Y %H:%M:%S'
        formatter = logging.Formatter(msgfmt, datefmt)
        # stdout
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(formatter)
        self.print_logger.addHandler(sh)
        # log file
        fh = logging.FileHandler(os.path.join(self.log_dir, 'log.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.print_logger.addHandler(fh)

        self.print_logger.info(f'Saving logs and checkpoints @ {self.logdir}')

    def log_param_num(self, model):
        param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.print_logger.info(f'The number of parameters: {param_num/1e+6:.3f} mil')
        self.add_scalar('num_params', param_num, 0)
        return param_num

    def log_iter_loss(self, loss, iter, phase):
        self.add_scalar(f'{phase}/loss_iter', loss, iter)

    def log_epoch_loss(self, loss, epoch, phase):
        self.add_scalar(f'{phase}/loss', loss, epoch)
        self.print_logger.info(f'{phase} ({epoch}): loss {loss:.3f};')

    def log_epoch_metrics(self, metrics_dict, epoch, phase):
        for metric, val in metrics_dict.items():
            self.add_scalar(f'{phase}/{metric}', val, epoch)
        metrics_dict = {k: round(v, 4) for k, v in metrics_dict.items()}
        self.print_logger.info(f'{phase} ({epoch}) metrics: {metrics_dict};')

    def log_test_metrics(self, metrics_dict, hparams_dict, best_epoch):
        allowed_types = (int, float, str, bool, torch.Tensor)
        hparams_dict = {k: v for k, v in hparams_dict.items() if isinstance(v, allowed_types)}
        metrics_dict = {f'test/{k}': round(v, 4) for k, v in metrics_dict.items()}
        exp, ssi, sei = summary.hparams(hparams_dict, metrics_dict)
        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metrics_dict.items():
            self.add_scalar(k, v, best_epoch)
        self.print_logger.info(f'test ({best_epoch}) metrics: {metrics_dict};')

    def log_best_model(self, model, loss, epoch, optimizer, metrics_dict):
        model_name = model.__class__.__name__
        self.best_model_path = os.path.join(self.logdir, f'{model_name}-{self.start_time}.pt')
        checkpoint = {
            'loss': loss,
            'metrics': metrics_dict,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
        }
        torch.save(checkpoint, self.best_model_path)
        self.print_logger.info(f'Saved model in {self.best_model_path}')
