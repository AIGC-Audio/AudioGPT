import matplotlib
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

matplotlib.use('Agg')
import glob
import itertools
import subprocess
import threading
import traceback

from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.callbacks import ModelCheckpoint

from functools import wraps
from torch.cuda._utils import _get_device_index
import numpy as np
import torch.optim
import torch.utils.data
import copy
import logging
import os
import re
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from torch.optim.optimizer import Optimizer


def get_a_var(obj):  # pragma: no cover
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def data_loader(fn):
    """
    Decorator to make any fx with this use the lazy property
    :param fn:
    :return:
    """

    wraps(fn)
    attr_name = '_lazy_' + fn.__name__

    def _get_data_loader(self):
        try:
            value = getattr(self, attr_name)
        except AttributeError:
            try:
                value = fn(self)  # Lazy evaluation, done only once.
                if (
                        value is not None and
                        not isinstance(value, list) and
                        fn.__name__ in ['test_dataloader', 'val_dataloader']
                ):
                    value = [value]
            except AttributeError as e:
                # Guard against AttributeError suppression. (Issue #142)
                traceback.print_exc()
                error = f'{fn.__name__}: An AttributeError was encountered: ' + str(e)
                raise RuntimeError(error) from e
            setattr(self, attr_name, value)  # Memoize evaluation.
        return value

    return _get_data_loader


def parallel_apply(modules, inputs, kwargs_tup=None, devices=None):  # pragma: no cover
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = list(map(lambda x: _get_device_index(x, True), devices))
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)

                # ---------------
                # CHANGE
                if module.training:
                    output = module.training_step(*input, **kwargs)

                elif module.testing:
                    output = module.test_step(*input, **kwargs)

                else:
                    output = module.validation_step(*input, **kwargs)
                # ---------------

            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    # make sure each module knows what training state it's in...
    # fixes weird bug where copies are out of sync
    root_m = modules[0]
    for m in modules[1:]:
        m.training = root_m.training
        m.testing = root_m.testing

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


def _find_tensors(obj):  # pragma: no cover
    r"""
    Recursively find all tensors contained in the specified object.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []


class DDP(DistributedDataParallel):
    """
    Override the forward call in lightning so it goes to training and validation step respectively
    """

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def forward(self, *inputs, **kwargs):  # pragma: no cover
        self._sync_params()
        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                # --------------
                # LIGHTNING MOD
                # --------------
                # normal
                # output = self.module(*inputs[0], **kwargs[0])
                # lightning
                if self.module.training:
                    output = self.module.training_step(*inputs[0], **kwargs[0])
                elif self.module.testing:
                    output = self.module.test_step(*inputs[0], **kwargs[0])
                else:
                    output = self.module.validation_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            # normal
            output = self.module(*inputs, **kwargs)

        if torch.is_grad_enabled():
            # We'll return the output object verbatim since it is a freeform
            # object. We need to find any tensors in this object, though,
            # because we need to figure out which parameters were used during
            # this forward pass, to ensure we short circuit reduction for any
            # unused parameters. Only if `find_unused_parameters` is set.
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        return output


class DP(DataParallel):
    """
    Override the forward call in lightning so it goes to training and validation step respectively
    """

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in itertools.chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            # lightning
            if self.module.training:
                return self.module.training_step(*inputs[0], **kwargs[0])
            elif self.module.testing:
                return self.module.test_step(*inputs[0], **kwargs[0])
            else:
                return self.module.validation_step(*inputs[0], **kwargs[0])

        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])


class GradientAccumulationScheduler:
    def __init__(self, scheduling: dict):
        if scheduling == {}:  # empty dict error
            raise TypeError("Empty dict cannot be interpreted correct")

        for key in scheduling.keys():
            if not isinstance(key, int) or not isinstance(scheduling[key], int):
                raise TypeError("All epoches and accumulation factor must be integers")

        minimal_epoch = min(scheduling.keys())
        if minimal_epoch < 1:
            msg = f"Epochs indexing from 1, epoch {minimal_epoch} cannot be interpreted correct"
            raise IndexError(msg)
        elif minimal_epoch != 1:  # if user didnt define first epoch accumulation factor
            scheduling.update({1: 1})

        self.scheduling = scheduling
        self.epochs = sorted(scheduling.keys())

    def on_epoch_begin(self, epoch, trainer):
        epoch += 1  # indexing epochs from 1
        for i in reversed(range(len(self.epochs))):
            if epoch >= self.epochs[i]:
                trainer.accumulate_grad_batches = self.scheduling.get(self.epochs[i])
                break


class LatestModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0, num_ckpt_keep=5,
                 save_weights_only=False, mode='auto', period=1, prefix='model', save_best=True):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        os.makedirs(filepath, exist_ok=True)
        self.num_ckpt_keep = num_ckpt_keep
        self.save_best = save_best
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_check = 0
        self.prefix = prefix
        self.best_k_models = {}
        # {filename: monitor}
        self.kth_best_model = ''
        self.save_top_k = 1
        self.task = None
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
            self.mode = 'min'
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
            self.mode = 'max'
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
                self.mode = 'max'
            else:
                self.monitor_op = np.less
                self.best = np.Inf
                self.mode = 'min'
        if os.path.exists(f'{self.filepath}/best_valid.npy'):
            self.best = np.load(f'{self.filepath}/best_valid.npy')[0]

    def get_all_ckpts(self):
        return sorted(glob.glob(f'{self.filepath}/{self.prefix}_ckpt_steps_*.ckpt'),
                      key=lambda x: -int(re.findall('.*steps\_(\d+)\.ckpt', x)[0]))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_check += 1
        best_filepath = f'{self.filepath}/{self.prefix}_ckpt_best.pt'
        if self.epochs_since_last_check >= self.period:
            self.epochs_since_last_check = 0
            filepath = f'{self.filepath}/{self.prefix}_ckpt_steps_{self.task.global_step}.ckpt'
            if self.verbose > 0:
                logging.info(f'Epoch {epoch:05d}@{self.task.global_step}: saving model to {filepath}')
            self._save_model(filepath)
            for old_ckpt in self.get_all_ckpts()[self.num_ckpt_keep:]:
                subprocess.check_call(f'rm -rf "{old_ckpt}"', shell=True)
                if self.verbose > 0:
                    logging.info(f'Delete ckpt: {os.path.basename(old_ckpt)}')
            current = logs.get(self.monitor)
            if current is not None and self.save_best:
                if self.monitor_op(current, self.best):
                    self.best = current
                    if self.verbose > 0:
                        logging.info(
                            f'Epoch {epoch:05d}@{self.task.global_step}: {self.monitor} reached'
                            f' {current:0.5f} (best {self.best:0.5f}), saving model to'
                            f' {best_filepath} as top 1')
                    self._save_model(best_filepath)
                    np.save(f'{self.filepath}/best_valid.npy', [self.best])


class BaseTrainer:
    def __init__(
            self,
            logger=True,
            checkpoint_callback=True,
            default_save_path=None,
            gradient_clip_val=0,
            process_position=0,
            gpus=-1,
            log_gpu_memory=None,
            show_progress_bar=True,
            track_grad_norm=-1,
            check_val_every_n_epoch=1,
            accumulate_grad_batches=1,
            max_updates=1000,
            min_epochs=1,
            val_check_interval=1.0,
            log_save_interval=100,
            row_log_interval=10,
            print_nan_grads=False,
            weights_summary='full',
            num_sanity_val_steps=5,
            resume_from_checkpoint=None,
    ):
        self.log_gpu_memory = log_gpu_memory
        self.gradient_clip_val = gradient_clip_val
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.track_grad_norm = track_grad_norm
        self.on_gpu = True if (gpus and torch.cuda.is_available()) else False
        self.process_position = process_position
        self.weights_summary = weights_summary
        self.max_updates = max_updates
        self.min_epochs = min_epochs
        self.num_sanity_val_steps = num_sanity_val_steps
        self.print_nan_grads = print_nan_grads
        self.resume_from_checkpoint = resume_from_checkpoint
        self.default_save_path = default_save_path

        # training bookeeping
        self.total_batch_idx = 0
        self.running_loss = []
        self.avg_loss = 0
        self.batch_idx = 0
        self.tqdm_metrics = {}
        self.callback_metrics = {}
        self.num_val_batches = 0
        self.num_training_batches = 0
        self.num_test_batches = 0
        self.get_train_dataloader = None
        self.get_test_dataloaders = None
        self.get_val_dataloaders = None
        self.is_iterable_train_dataloader = False

        # training state
        self.model = None
        self.testing = False
        self.disable_validation = False
        self.lr_schedulers = []
        self.optimizers = None
        self.global_step = 0
        self.current_epoch = 0
        self.total_batches = 0

        # configure checkpoint callback
        self.checkpoint_callback = checkpoint_callback
        self.checkpoint_callback.save_function = self.save_checkpoint
        self.weights_save_path = self.checkpoint_callback.filepath

        # accumulated grads
        self.configure_accumulated_gradients(accumulate_grad_batches)

        # allow int, string and gpu list
        self.data_parallel_device_ids = [
            int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x != '']
        if len(self.data_parallel_device_ids) == 0:
            self.root_gpu = None
            self.on_gpu = False
        else:
            self.root_gpu = self.data_parallel_device_ids[0]
            self.on_gpu = True

        # distributed backend choice
        self.use_ddp = False
        self.use_dp = False
        self.single_gpu = False
        self.distributed_backend = 'ddp' if self.num_gpus > 0 else 'dp'
        self.set_distributed_mode(self.distributed_backend)

        self.proc_rank = 0
        self.world_size = 1
        self.node_rank = 0

        # can't init progress bar here because starting a new process
        # means the progress_bar won't survive pickling
        self.show_progress_bar = show_progress_bar

        # logging
        self.log_save_interval = log_save_interval
        self.val_check_interval = val_check_interval
        self.logger = logger
        self.logger.rank = 0
        self.row_log_interval = row_log_interval

    @property
    def num_gpus(self):
        gpus = self.data_parallel_device_ids
        if gpus is None:
            return 0
        else:
            return len(gpus)

    @property
    def data_parallel(self):
        return self.use_dp or self.use_ddp

    def get_model(self):
        is_dp_module = isinstance(self.model, (DDP, DP))
        model = self.model.module if is_dp_module else self.model
        return model

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    def fit(self, model):
        if self.use_ddp:
            mp.spawn(self.ddp_train, nprocs=self.num_gpus, args=(model,))
        else:
            model.model = model.build_model()
            if not self.testing:
                self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())
            if self.use_dp:
                model.cuda(self.root_gpu)
                model = DP(model, device_ids=self.data_parallel_device_ids)
            elif self.single_gpu:
                model.cuda(self.root_gpu)
            self.run_pretrain_routine(model)
        return 1

    def init_optimizers(self, optimizers):

        # single optimizer
        if isinstance(optimizers, Optimizer):
            return [optimizers], []

        # two lists
        elif len(optimizers) == 2 and isinstance(optimizers[0], list):
            optimizers, lr_schedulers = optimizers
            return optimizers, lr_schedulers

        # single list or tuple
        elif isinstance(optimizers, list) or isinstance(optimizers, tuple):
            return optimizers, []

    def run_pretrain_routine(self, model):
        """Sanity check a few things before starting actual training.

        :param model:
        """
        ref_model = model
        if self.data_parallel:
            ref_model = model.module

        # give model convenience properties
        ref_model.trainer = self

        # set local properties on the model
        self.copy_trainer_model_properties(ref_model)

        # link up experiment object
        if self.logger is not None:
            ref_model.logger = self.logger
            self.logger.save()

        if self.use_ddp:
            dist.barrier()

        # set up checkpoint callback
        # self.configure_checkpoint_callback()

        # transfer data loaders from model
        self.get_dataloaders(ref_model)

        # track model now.
        # if cluster resets state, the model will update with the saved weights
        self.model = model

        # restore training and model before hpc call
        self.restore_weights(model)

        # when testing requested only run test and return
        if self.testing:
            self.run_evaluation(test=True)
            return

        # check if we should run validation during training
        self.disable_validation = self.num_val_batches == 0

        # run tiny validation (if validation defined)
        # to make sure program won't crash during val
        ref_model.on_sanity_check_start()
        ref_model.on_train_start()
        if not self.disable_validation and self.num_sanity_val_steps > 0:
            # init progress bars for validation sanity check
            pbar = tqdm.tqdm(desc='Validation sanity check',
                             total=self.num_sanity_val_steps * len(self.get_val_dataloaders()),
                             leave=False, position=2 * self.process_position,
                             disable=not self.show_progress_bar, dynamic_ncols=True, unit='batch')
            self.main_progress_bar = pbar
            # dummy validation progress bar
            self.val_progress_bar = tqdm.tqdm(disable=True)

            self.evaluate(model, self.get_val_dataloaders(), self.num_sanity_val_steps, self.testing)

            # close progress bars
            self.main_progress_bar.close()
            self.val_progress_bar.close()

        # init progress bar
        pbar = tqdm.tqdm(leave=True, position=2 * self.process_position,
                         disable=not self.show_progress_bar, dynamic_ncols=True, unit='batch',
                         file=sys.stdout)
        self.main_progress_bar = pbar

        # clear cache before training
        if self.on_gpu:
            torch.cuda.empty_cache()

        # CORE TRAINING LOOP
        self.train()

    def test(self, model):
        self.testing = True
        self.fit(model)

    @property
    def training_tqdm_dict(self):
        tqdm_dict = {
            'step': '{}'.format(self.global_step),
        }
        tqdm_dict.update(self.tqdm_metrics)
        return tqdm_dict

    # --------------------
    # restore ckpt
    # --------------------
    def restore_weights(self, model):
        """
        To restore weights we have two cases.
        First, attempt to restore hpc weights. If successful, don't restore
        other weights.

        Otherwise, try to restore actual weights
        :param model:
        :return:
        """
        # clear cache before restore
        if self.on_gpu:
            torch.cuda.empty_cache()

        if self.resume_from_checkpoint is not None:
            self.restore(self.resume_from_checkpoint, on_gpu=self.on_gpu)
        else:
            # restore weights if same exp version
            self.restore_state_if_checkpoint_exists(model)

        # wait for all models to restore weights
        if self.use_ddp:
            # wait for all processes to catch up
            dist.barrier()

        # clear cache after restore
        if self.on_gpu:
            torch.cuda.empty_cache()

    def restore_state_if_checkpoint_exists(self, model):
        did_restore = False

        # do nothing if there's not dir or callback
        no_ckpt_callback = (self.checkpoint_callback is None) or (not self.checkpoint_callback)
        if no_ckpt_callback or not os.path.exists(self.checkpoint_callback.filepath):
            return did_restore

        # restore trainer state and model if there is a weight for this experiment
        last_steps = -1
        last_ckpt_name = None

        # find last epoch
        checkpoints = os.listdir(self.checkpoint_callback.filepath)
        for name in checkpoints:
            if '.ckpt' in name and not name.endswith('part'):
                if 'steps_' in name:
                    steps = name.split('steps_')[1]
                    steps = int(re.sub('[^0-9]', '', steps))

                    if steps > last_steps:
                        last_steps = steps
                        last_ckpt_name = name

        # restore last checkpoint
        if last_ckpt_name is not None:
            last_ckpt_path = os.path.join(self.checkpoint_callback.filepath, last_ckpt_name)
            self.restore(last_ckpt_path, self.on_gpu)
            logging.info(f'model and trainer restored from checkpoint: {last_ckpt_path}')
            did_restore = True

        return did_restore

    def restore(self, checkpoint_path, on_gpu):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # load model state
        model = self.get_model()

        # load the state_dict on the model automatically
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if on_gpu:
            model.cuda(self.root_gpu)
        # load training state (affects trainer only)
        self.restore_training_state(checkpoint)
        model.global_step = self.global_step
        del checkpoint

        try:
            if dist.is_initialized() and dist.get_rank() > 0:
                return
        except Exception as e:
            print(e)
            return

    def restore_training_state(self, checkpoint):
        """
        Restore trainer state.
        Model will get its change to update
        :param checkpoint:
        :return:
        """
        if self.checkpoint_callback is not None and self.checkpoint_callback is not False:
            self.checkpoint_callback.best = checkpoint['checkpoint_callback_best']

        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']

        if self.testing:
            return

        # restore the optimizers
        optimizer_states = checkpoint['optimizer_states']
        for optimizer, opt_state in zip(self.optimizers, optimizer_states):
            if optimizer is None:
                return
            optimizer.load_state_dict(opt_state)

            # move optimizer to GPU 1 weight at a time
            # avoids OOM
            if self.root_gpu is not None:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.root_gpu)

        # restore the lr schedulers
        lr_schedulers = checkpoint['lr_schedulers']
        for scheduler, lrs_state in zip(self.lr_schedulers, lr_schedulers):
            scheduler.load_state_dict(lrs_state)

    # --------------------
    # MODEL SAVE CHECKPOINT
    # --------------------
    def _atomic_save(self, checkpoint, filepath):
        """Saves a checkpoint atomically, avoiding the creation of incomplete checkpoints.

        This will create a temporary checkpoint with a suffix of ``.part``, then copy it to the final location once
        saving is finished.

        Args:
            checkpoint (object): The object to save.
                Built to be used with the ``dump_checkpoint`` method, but can deal with anything which ``torch.save``
                accepts.
            filepath (str|pathlib.Path): The path to which the checkpoint will be saved.
                This points to the file that the checkpoint will be stored in.
        """
        tmp_path = str(filepath) + ".part"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, filepath)

    def save_checkpoint(self, filepath):
        checkpoint = self.dump_checkpoint()
        self._atomic_save(checkpoint, filepath)

    def dump_checkpoint(self):

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step
        }

        if self.checkpoint_callback is not None and self.checkpoint_callback is not False:
            checkpoint['checkpoint_callback_best'] = self.checkpoint_callback.best

        # save optimizers
        optimizer_states = []
        for i, optimizer in enumerate(self.optimizers):
            if optimizer is not None:
                optimizer_states.append(optimizer.state_dict())

        checkpoint['optimizer_states'] = optimizer_states

        # save lr schedulers
        lr_schedulers = []
        for i, scheduler in enumerate(self.lr_schedulers):
            lr_schedulers.append(scheduler.state_dict())

        checkpoint['lr_schedulers'] = lr_schedulers

        # add the hparams and state_dict from the model
        model = self.get_model()
        checkpoint['state_dict'] = model.state_dict()
        # give the model a chance to add a few things
        model.on_save_checkpoint(checkpoint)

        return checkpoint

    def copy_trainer_model_properties(self, model):
        if isinstance(model, DP):
            ref_model = model.module
        elif isinstance(model, DDP):
            ref_model = model.module
        else:
            ref_model = model

        for m in [model, ref_model]:
            m.trainer = self
            m.on_gpu = self.on_gpu
            m.use_dp = self.use_dp
            m.use_ddp = self.use_ddp
            m.testing = self.testing
            m.single_gpu = self.single_gpu

    def transfer_batch_to_gpu(self, batch, gpu_id):
        # base case: object can be directly moved using `cuda` or `to`
        if callable(getattr(batch, 'cuda', None)):
            return batch.cuda(gpu_id, non_blocking=True)

        elif callable(getattr(batch, 'to', None)):
            return batch.to(torch.device('cuda', gpu_id), non_blocking=True)

        # when list
        elif isinstance(batch, list):
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id)
            return batch

        # when tuple
        elif isinstance(batch, tuple):
            batch = list(batch)
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id)
            return tuple(batch)

        # when dict
        elif isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = self.transfer_batch_to_gpu(v, gpu_id)

            return batch

        # nothing matches, return the value as is without transform
        return batch

    def set_distributed_mode(self, distributed_backend):
        # skip for CPU
        if self.num_gpus == 0:
            return

        # single GPU case
        # in single gpu case we allow ddp so we can train on multiple
        # nodes, 1 gpu per node
        elif self.num_gpus == 1:
            self.single_gpu = True
            self.use_dp = False
            self.use_ddp = False
            self.root_gpu = 0
            self.data_parallel_device_ids = [0]
        else:
            if distributed_backend is not None:
                self.use_dp = distributed_backend == 'dp'
                self.use_ddp = distributed_backend == 'ddp'
            elif distributed_backend is None:
                self.use_dp = True
                self.use_ddp = False

        logging.info(f'gpu available: {torch.cuda.is_available()}, used: {self.on_gpu}')

    def ddp_train(self, gpu_idx, model):
        """
        Entry point into a DP thread
        :param gpu_idx:
        :param model:
        :param cluster_obj:
        :return:
        """
        # otherwise default to node rank 0
        self.node_rank = 0

        # show progressbar only on progress_rank 0
        self.show_progress_bar = self.show_progress_bar and self.node_rank == 0 and gpu_idx == 0

        # determine which process we are and world size
        if self.use_ddp:
            self.proc_rank = self.node_rank * self.num_gpus + gpu_idx
            self.world_size = self.num_gpus

        # let the exp know the rank to avoid overwriting logs
        if self.logger is not None:
            self.logger.rank = self.proc_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        model.trainer = self
        model.init_ddp_connection(self.proc_rank, self.world_size)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        model.model = model.build_model()
        if not self.testing:
            self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

        # MODEL
        # copy model to each gpu
        if self.distributed_backend == 'ddp':
            torch.cuda.set_device(gpu_idx)
        model.cuda(gpu_idx)

        # set model properties before going into wrapper
        self.copy_trainer_model_properties(model)

        # override root GPU
        self.root_gpu = gpu_idx

        if self.distributed_backend == 'ddp':
            device_ids = [gpu_idx]
        else:
            device_ids = None

        # allow user to configure ddp
        model = model.configure_ddp(model, device_ids)

        # continue training routine
        self.run_pretrain_routine(model)

    def resolve_root_node_address(self, root_node):
        if '[' in root_node:
            name = root_node.split('[')[0]
            number = root_node.split(',')[0]
            if '-' in number:
                number = number.split('-')[0]

            number = re.sub('[^0-9]', '', number)
            root_node = name + number

        return root_node

    def log_metrics(self, metrics, grad_norm_dic, step=None):
        """Logs the metric dict passed in.

        :param metrics:
        :param grad_norm_dic:
        """
        # added metrics by Lightning for convenience
        metrics['epoch'] = self.current_epoch

        # add norms
        metrics.update(grad_norm_dic)

        # turn all tensors to scalars
        scalar_metrics = self.metrics_to_scalars(metrics)

        step = step if step is not None else self.global_step
        # log actual metrics
        if self.proc_rank == 0 and self.logger is not None:
            self.logger.log_metrics(scalar_metrics, step=step)
            self.logger.save()

    def add_tqdm_metrics(self, metrics):
        for k, v in metrics.items():
            if type(v) is torch.Tensor:
                v = v.item()

            self.tqdm_metrics[k] = v

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    def process_output(self, output, train=False):
        """Reduces output according to the training mode.

        Separates loss from logging and tqdm metrics
        :param output:
        :return:
        """
        # ---------------
        # EXTRACT CALLBACK KEYS
        # ---------------
        # all keys not progress_bar or log are candidates for callbacks
        callback_metrics = {}
        for k, v in output.items():
            if k not in ['progress_bar', 'log', 'hiddens']:
                callback_metrics[k] = v

        if train and self.use_dp:
            num_gpus = self.num_gpus
            callback_metrics = self.reduce_distributed_output(callback_metrics, num_gpus)

        for k, v in callback_metrics.items():
            if isinstance(v, torch.Tensor):
                callback_metrics[k] = v.item()

        # ---------------
        # EXTRACT PROGRESS BAR KEYS
        # ---------------
        try:
            progress_output = output['progress_bar']

            # reduce progress metrics for tqdm when using dp
            if train and self.use_dp:
                num_gpus = self.num_gpus
                progress_output = self.reduce_distributed_output(progress_output, num_gpus)

            progress_bar_metrics = progress_output
        except Exception:
            progress_bar_metrics = {}

        # ---------------
        # EXTRACT LOGGING KEYS
        # ---------------
        # extract metrics to log to experiment
        try:
            log_output = output['log']

            # reduce progress metrics for tqdm when using dp
            if train and self.use_dp:
                num_gpus = self.num_gpus
                log_output = self.reduce_distributed_output(log_output, num_gpus)

            log_metrics = log_output
        except Exception:
            log_metrics = {}

        # ---------------
        # EXTRACT LOSS
        # ---------------
        # if output dict doesn't have the keyword loss
        # then assume the output=loss if scalar
        loss = None
        if train:
            try:
                loss = output['loss']
            except Exception:
                if type(output) is torch.Tensor:
                    loss = output
                else:
                    raise RuntimeError(
                        'No `loss` value in the dictionary returned from `model.training_step()`.'
                    )

            # when using dp need to reduce the loss
            if self.use_dp:
                loss = self.reduce_distributed_output(loss, self.num_gpus)

        # ---------------
        # EXTRACT HIDDEN
        # ---------------
        hiddens = output.get('hiddens')

        # use every metric passed in as a candidate for callback
        callback_metrics.update(progress_bar_metrics)
        callback_metrics.update(log_metrics)

        # convert tensors to numpy
        for k, v in callback_metrics.items():
            if isinstance(v, torch.Tensor):
                callback_metrics[k] = v.item()

        return loss, progress_bar_metrics, log_metrics, callback_metrics, hiddens

    def reduce_distributed_output(self, output, num_gpus):
        if num_gpus <= 1:
            return output

        # when using DP, we get one output per gpu
        # average outputs and return
        if type(output) is torch.Tensor:
            return output.mean()

        for k, v in output.items():
            # recurse on nested dics
            if isinstance(output[k], dict):
                output[k] = self.reduce_distributed_output(output[k], num_gpus)

            # do nothing when there's a scalar
            elif isinstance(output[k], torch.Tensor) and output[k].dim() == 0:
                pass

            # reduce only metrics that have the same number of gpus
            elif output[k].size(0) == num_gpus:
                reduced = torch.mean(output[k])
                output[k] = reduced
        return output

    def clip_gradients(self):
        if self.gradient_clip_val > 0:
            model = self.get_model()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)

    def print_nan_gradients(self):
        model = self.get_model()
        for param in model.parameters():
            if (param.grad is not None) and torch.isnan(param.grad.float()).any():
                logging.info(param, param.grad)

    def configure_accumulated_gradients(self, accumulate_grad_batches):
        self.accumulate_grad_batches = None

        if isinstance(accumulate_grad_batches, dict):
            self.accumulation_scheduler = GradientAccumulationScheduler(accumulate_grad_batches)
        elif isinstance(accumulate_grad_batches, int):
            schedule = {1: accumulate_grad_batches}
            self.accumulation_scheduler = GradientAccumulationScheduler(schedule)
        else:
            raise TypeError("Gradient accumulation supports only int and dict types")

    def get_dataloaders(self, model):
        if not self.testing:
            self.init_train_dataloader(model)
            self.init_val_dataloader(model)
        else:
            self.init_test_dataloader(model)

        if self.use_ddp:
            dist.barrier()
            if not self.testing:
                self.get_train_dataloader()
                self.get_val_dataloaders()
            else:
                self.get_test_dataloaders()

    def init_train_dataloader(self, model):
        self.fisrt_epoch = True
        self.get_train_dataloader = model.train_dataloader
        if isinstance(self.get_train_dataloader(), torch.utils.data.DataLoader):
            self.num_training_batches = len(self.get_train_dataloader())
            self.num_training_batches = int(self.num_training_batches)
        else:
            self.num_training_batches = float('inf')
            self.is_iterable_train_dataloader = True
        if isinstance(self.val_check_interval, int):
            self.val_check_batch = self.val_check_interval
        else:
            self._percent_range_check('val_check_interval')
            self.val_check_batch = int(self.num_training_batches * self.val_check_interval)
            self.val_check_batch = max(1, self.val_check_batch)

    def init_val_dataloader(self, model):
        self.get_val_dataloaders = model.val_dataloader
        self.num_val_batches = 0
        if self.get_val_dataloaders() is not None:
            if isinstance(self.get_val_dataloaders()[0], torch.utils.data.DataLoader):
                self.num_val_batches = sum(len(dataloader) for dataloader in self.get_val_dataloaders())
                self.num_val_batches = int(self.num_val_batches)
            else:
                self.num_val_batches = float('inf')

    def init_test_dataloader(self, model):
        self.get_test_dataloaders = model.test_dataloader
        if self.get_test_dataloaders() is not None:
            if isinstance(self.get_test_dataloaders()[0], torch.utils.data.DataLoader):
                self.num_test_batches = sum(len(dataloader) for dataloader in self.get_test_dataloaders())
                self.num_test_batches = int(self.num_test_batches)
            else:
                self.num_test_batches = float('inf')

    def evaluate(self, model, dataloaders, max_batches, test=False):
        """Run evaluation code.

        :param model: PT model
        :param dataloaders: list of PT dataloaders
        :param max_batches: Scalar
        :param test: boolean
        :return:
        """
        # enable eval mode
        model.zero_grad()
        model.eval()

        # copy properties for forward overrides
        self.copy_trainer_model_properties(model)

        # disable gradients to save memory
        torch.set_grad_enabled(False)

        if test:
            self.get_model().test_start()
        # bookkeeping
        outputs = []

        # run training
        for dataloader_idx, dataloader in enumerate(dataloaders):
            dl_outputs = []
            for batch_idx, batch in enumerate(dataloader):

                if batch is None:  # pragma: no cover
                    continue

                # stop short when on fast_dev_run (sets max_batch=1)
                if batch_idx >= max_batches:
                    break

                # -----------------
                # RUN EVALUATION STEP
                # -----------------
                output = self.evaluation_forward(model,
                                                 batch,
                                                 batch_idx,
                                                 dataloader_idx,
                                                 test)

                # track outputs for collation
                dl_outputs.append(output)

                # batch done
                if test:
                    self.test_progress_bar.update(1)
                else:
                    self.val_progress_bar.update(1)
            outputs.append(dl_outputs)

        # with a single dataloader don't pass an array
        if len(dataloaders) == 1:
            outputs = outputs[0]

        # give model a chance to do something with the outputs (and method defined)
        model = self.get_model()
        if test:
            eval_results_ = model.test_end(outputs)
        else:
            eval_results_ = model.validation_end(outputs)
        eval_results = eval_results_

        # enable train mode again
        model.train()

        # enable gradients to save memory
        torch.set_grad_enabled(True)

        return eval_results

    def run_evaluation(self, test=False):
        # when testing make sure user defined a test step
        model = self.get_model()
        model.on_pre_performance_check()

        # select dataloaders
        if test:
            dataloaders = self.get_test_dataloaders()
            max_batches = self.num_test_batches
        else:
            # val
            dataloaders = self.get_val_dataloaders()
            max_batches = self.num_val_batches

        # init validation or test progress bar
        # main progress bar will already be closed when testing so initial position is free
        position = 2 * self.process_position + (not test)
        desc = 'Testing' if test else 'Validating'
        pbar = tqdm.tqdm(desc=desc, total=max_batches, leave=test, position=position,
                         disable=not self.show_progress_bar, dynamic_ncols=True,
                         unit='batch', file=sys.stdout)
        setattr(self, f'{"test" if test else "val"}_progress_bar', pbar)

        # run evaluation
        eval_results = self.evaluate(self.model,
                                     dataloaders,
                                     max_batches,
                                     test)
        if eval_results is not None:
            _, prog_bar_metrics, log_metrics, callback_metrics, _ = self.process_output(
                eval_results)

            # add metrics to prog bar
            self.add_tqdm_metrics(prog_bar_metrics)

            # log metrics
            self.log_metrics(log_metrics, {})

            # track metrics for callbacks
            self.callback_metrics.update(callback_metrics)

        # hook
        model.on_post_performance_check()

        # add model specific metrics
        tqdm_metrics = self.training_tqdm_dict
        if not test:
            self.main_progress_bar.set_postfix(**tqdm_metrics)

        # close progress bar
        if test:
            self.test_progress_bar.close()
        else:
            self.val_progress_bar.close()

        # model checkpointing
        if self.proc_rank == 0 and self.checkpoint_callback is not None and not test:
            self.checkpoint_callback.on_epoch_end(epoch=self.current_epoch,
                                                  logs=self.callback_metrics)

    def evaluation_forward(self, model, batch, batch_idx, dataloader_idx, test=False):
        # make dataloader_idx arg in validation_step optional
        args = [batch, batch_idx]

        if test and len(self.get_test_dataloaders()) > 1:
            args.append(dataloader_idx)

        elif not test and len(self.get_val_dataloaders()) > 1:
            args.append(dataloader_idx)

        # handle DP, DDP forward
        if self.use_ddp or self.use_dp:
            output = model(*args)
            return output

        # single GPU
        if self.single_gpu:
            # for single GPU put inputs on gpu manually
            root_gpu = 0
            if isinstance(self.data_parallel_device_ids, list):
                root_gpu = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(batch, root_gpu)
            args[0] = batch

        # CPU
        if test:
            output = model.test_step(*args)
        else:
            output = model.validation_step(*args)

        return output

    def train(self):
        model = self.get_model()
        # run all epochs
        for epoch in range(self.current_epoch, 1000000):
            # set seed for distributed sampler (enables shuffling for each epoch)
            if self.use_ddp and hasattr(self.get_train_dataloader().sampler, 'set_epoch'):
                self.get_train_dataloader().sampler.set_epoch(epoch)

            # get model
            model = self.get_model()

            # update training progress in trainer and model
            model.current_epoch = epoch
            self.current_epoch = epoch

            total_val_batches = 0
            if not self.disable_validation:
                # val can be checked multiple times in epoch
                is_val_epoch = (self.current_epoch + 1) % self.check_val_every_n_epoch == 0
                val_checks_per_epoch = self.num_training_batches // self.val_check_batch
                val_checks_per_epoch = val_checks_per_epoch if is_val_epoch else 0
                total_val_batches = self.num_val_batches * val_checks_per_epoch

            # total batches includes multiple val checks
            self.total_batches = self.num_training_batches + total_val_batches
            self.batch_loss_value = 0  # accumulated grads

            if self.is_iterable_train_dataloader:
                # for iterable train loader, the progress bar never ends
                num_iterations = None
            else:
                num_iterations = self.total_batches

            # reset progress bar
            # .reset() doesn't work on disabled progress bar so we should check
            desc = f'Epoch {epoch + 1}' if not self.is_iterable_train_dataloader else ''
            self.main_progress_bar.set_description(desc)

            # changing gradient according accumulation_scheduler
            self.accumulation_scheduler.on_epoch_begin(epoch, self)

            # -----------------
            # RUN TNG EPOCH
            # -----------------
            self.run_training_epoch()

            # update LR schedulers
            if self.lr_schedulers is not None:
                for lr_scheduler in self.lr_schedulers:
                    lr_scheduler.step(epoch=self.current_epoch)

        self.main_progress_bar.close()

        model.on_train_end()

        if self.logger is not None:
            self.logger.finalize("success")

    def run_training_epoch(self):
        # before epoch hook
        if self.is_function_implemented('on_epoch_start'):
            model = self.get_model()
            model.on_epoch_start()

        # run epoch
        for batch_idx, batch in enumerate(self.get_train_dataloader()):
            # stop epoch if we limited the number of training batches
            if batch_idx >= self.num_training_batches:
                break

            self.batch_idx = batch_idx

            model = self.get_model()
            model.global_step = self.global_step

            # ---------------
            # RUN TRAIN STEP
            # ---------------
            output = self.run_training_batch(batch, batch_idx)
            batch_result, grad_norm_dic, batch_step_metrics = output

            # when returning -1 from train_step, we end epoch early
            early_stop_epoch = batch_result == -1

            # ---------------
            # RUN VAL STEP
            # ---------------
            should_check_val = (
                    not self.disable_validation and self.global_step % self.val_check_batch == 0 and not self.fisrt_epoch)
            self.fisrt_epoch = False

            if should_check_val:
                self.run_evaluation(test=self.testing)

            # when logs should be saved
            should_save_log = (batch_idx + 1) % self.log_save_interval == 0 or early_stop_epoch
            if should_save_log:
                if self.proc_rank == 0 and self.logger is not None:
                    self.logger.save()

            # when metrics should be logged
            should_log_metrics = batch_idx % self.row_log_interval == 0 or early_stop_epoch
            if should_log_metrics:
                # logs user requested information to logger
                self.log_metrics(batch_step_metrics, grad_norm_dic)

            self.global_step += 1
            self.total_batch_idx += 1

            # end epoch early
            # stop when the flag is changed or we've gone past the amount
            # requested in the batches
            if early_stop_epoch:
                break
            if self.global_step > self.max_updates:
                print("| Training end..")
                exit()

        # epoch end hook
        if self.is_function_implemented('on_epoch_end'):
            model = self.get_model()
            model.on_epoch_end()

    def run_training_batch(self, batch, batch_idx):
        # track grad norms
        grad_norm_dic = {}

        # track all metrics for callbacks
        all_callback_metrics = []

        # track metrics to log
        all_log_metrics = []

        if batch is None:
            return 0, grad_norm_dic, {}

        # hook
        if self.is_function_implemented('on_batch_start'):
            model_ref = self.get_model()
            response = model_ref.on_batch_start(batch)

            if response == -1:
                return -1, grad_norm_dic, {}

        splits = [batch]
        self.hiddens = None
        for split_idx, split_batch in enumerate(splits):
            self.split_idx = split_idx

            # call training_step once per optimizer
            for opt_idx, optimizer in enumerate(self.optimizers):
                if optimizer is None:
                    continue
                # make sure only the gradients of the current optimizer's paramaters are calculated
                # in the training step to prevent dangling gradients in multiple-optimizer setup.
                if len(self.optimizers) > 1:
                    for param in self.get_model().parameters():
                        param.requires_grad = False
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.requires_grad = True

                # wrap the forward step in a closure so second order methods work
                def optimizer_closure():
                    # forward pass
                    output = self.training_forward(
                        split_batch, batch_idx, opt_idx, self.hiddens)

                    closure_loss = output[0]
                    progress_bar_metrics = output[1]
                    log_metrics = output[2]
                    callback_metrics = output[3]
                    self.hiddens = output[4]
                    if closure_loss is None:
                        return None

                    # accumulate loss
                    # (if accumulate_grad_batches = 1 no effect)
                    closure_loss = closure_loss / self.accumulate_grad_batches

                    # backward pass
                    model_ref = self.get_model()
                    if closure_loss.requires_grad:
                        model_ref.backward(closure_loss, optimizer)

                    # track metrics for callbacks
                    all_callback_metrics.append(callback_metrics)

                    # track progress bar metrics
                    self.add_tqdm_metrics(progress_bar_metrics)
                    all_log_metrics.append(log_metrics)

                    # insert after step hook
                    if self.is_function_implemented('on_after_backward'):
                        model_ref = self.get_model()
                        model_ref.on_after_backward()

                    return closure_loss

                # calculate loss
                loss = optimizer_closure()
                if loss is None:
                    continue

                # nan grads
                if self.print_nan_grads:
                    self.print_nan_gradients()

                # track total loss for logging (avoid mem leaks)
                self.batch_loss_value += loss.item()

                # gradient update with accumulated gradients
                if (self.batch_idx + 1) % self.accumulate_grad_batches == 0:

                    # track gradient norms when requested
                    if batch_idx % self.row_log_interval == 0:
                        if self.track_grad_norm > 0:
                            model = self.get_model()
                            grad_norm_dic = model.grad_norm(
                                self.track_grad_norm)

                    # clip gradients
                    self.clip_gradients()

                    # calls .step(), .zero_grad()
                    # override function to modify this behavior
                    model = self.get_model()
                    model.optimizer_step(self.current_epoch, batch_idx, optimizer, opt_idx)

                    # calculate running loss for display
                    self.running_loss.append(self.batch_loss_value)
                    self.batch_loss_value = 0
                    self.avg_loss = np.mean(self.running_loss[-100:])

        # activate batch end hook
        if self.is_function_implemented('on_batch_end'):
            model = self.get_model()
            model.on_batch_end()

        # update progress bar
        self.main_progress_bar.update(1)
        self.main_progress_bar.set_postfix(**self.training_tqdm_dict)

        # collapse all metrics into one dict
        all_log_metrics = {k: v for d in all_log_metrics for k, v in d.items()}

        # track all metrics for callbacks
        self.callback_metrics.update({k: v for d in all_callback_metrics for k, v in d.items()})

        return 0, grad_norm_dic, all_log_metrics

    def training_forward(self, batch, batch_idx, opt_idx, hiddens):
        """
        Handle forward for each training case (distributed, single gpu, etc...)
        :param batch:
        :param batch_idx:
        :return:
        """
        # ---------------
        # FORWARD
        # ---------------
        # enable not needing to add opt_idx to training_step
        args = [batch, batch_idx, opt_idx]

        # distributed forward
        if self.use_ddp or self.use_dp:
            output = self.model(*args)
        # single GPU forward
        elif self.single_gpu:
            gpu_id = 0
            if isinstance(self.data_parallel_device_ids, list):
                gpu_id = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(copy.copy(batch), gpu_id)
            args[0] = batch
            output = self.model.training_step(*args)
        # CPU forward
        else:
            output = self.model.training_step(*args)

        # allow any mode to define training_end
        model_ref = self.get_model()
        output_ = model_ref.training_end(output)
        if output_ is not None:
            output = output_

        # format and reduce outputs accordingly
        output = self.process_output(output, train=True)

        return output

    # ---------------
    # Utils
    # ---------------
    def is_function_implemented(self, f_name):
        model = self.get_model()
        f_op = getattr(model, f_name, None)
        return callable(f_op)

    def _percent_range_check(self, name):
        value = getattr(self, name)
        msg = f"`{name}` must lie in the range [0.0, 1.0], but got {value:.3f}."
        if name == "val_check_interval":
            msg += " If you want to disable validation set `val_percent_check` to 0.0 instead."

        if not 0. <= value <= 1.:
            raise ValueError(msg)
