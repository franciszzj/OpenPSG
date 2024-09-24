from typing import Callable, Dict, List, Optional, Tuple, Union
import io
import os.path as osp
import platform
import shutil
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Optimizer

import mmcv
from mmcv.runner import HOOKS, CheckpointHook, master_only
from mmcv.runner.checkpoint import get_state_dict, weights_to_cpu
from mmcv.fileio import FileClient
from mmcv.parallel.utils import is_module_wrapper


@HOOKS.register_module()
class PartCheckpointHook(CheckpointHook):

    @master_only
    def _save_checkpoint(self, runner):
        """Save the current checkpoint and delete unwanted checkpoint."""
        # runner.save_checkpoint(
        #     self.out_dir, save_optimizer=self.save_optimizer, **self.args)
        if self.by_epoch:
            epoch_based_save_checkpoint(
                self.out_dir, self.args.get(
                    'filename_tmpl', 'epoch_{}.pth'), self.save_optimizer,
                runner.meta, self.args.get('create_symlink', True),
                runner.meta, runner.epoch, runner.iter, runner.optimizer, runner.model)
        else:
            assert self.by_epoch, 'Only epoch based checkpoint is supported.'
        if runner.meta is not None:
            if self.by_epoch:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'epoch_{}.pth').format(runner.epoch + 1)
            else:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'iter_{}.pth').format(runner.iter + 1)
            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs']['last_ckpt'] = self.file_client.join_path(
                self.out_dir, cur_ckpt_filename)
        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            if self.by_epoch:
                name = 'epoch_{}.pth'
                current_ckpt = runner.epoch + 1
            else:
                name = 'iter_{}.pth'
                current_ckpt = runner.iter + 1
            redundant_ckpts = range(
                current_ckpt - self.max_keep_ckpts * self.interval, 0,
                -self.interval)
            filename_tmpl = self.args.get('filename_tmpl', name)
            for _step in redundant_ckpts:
                ckpt_path = self.file_client.join_path(
                    self.out_dir, filename_tmpl.format(_step))
                if self.file_client.isfile(ckpt_path):
                    self.file_client.remove(ckpt_path)
                else:
                    break


def epoch_based_save_checkpoint(out_dir, filename_tmpl, save_optimizer, meta, create_symlink,
                                self_meta, self_epoch, self_iter, self_optimizer, self_model):
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(
            f'meta should be a dict or None, but got {type(meta)}')
    if self_meta is not None:
        meta.update(self_meta)
        # Note: meta.update(self.meta) should be done before
        # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
        # there will be problems with resumed checkpoints.
        # More details in https://github.com/open-mmlab/mmcv/pull/1108
    meta.update(epoch=self_epoch + 1, iter=self_iter)

    filename = filename_tmpl.format(self_epoch + 1)
    filepath = osp.join(out_dir, filename)
    optimizer = self_optimizer if save_optimizer else None
    save_checkpoint(self_model, filepath, optimizer=optimizer, meta=meta)
    # in some environments, `os.symlink` is not supported, you may need to
    # set `create_symlink` to False
    if create_symlink:
        dst_file = osp.join(out_dir, 'latest.pth')
        if platform.system() != 'Windows':
            mmcv.symlink(filename, dst_file)
        else:
            shutil.copy(filepath, dst_file)


def keep_part_model(state_dict: OrderedDict, freeze_layers=[]) -> OrderedDict:
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_keep = OrderedDict()
    for key, val in state_dict.items():
        keep = True
        for not_keep in freeze_layers:
            if key.startswith(not_keep):
                keep = False
        if keep:
            state_dict_keep[key] = val
    # Keep metadata in state_dict
    state_dict_keep._metadata = getattr(  # type: ignore
        state_dict, '_metadata', OrderedDict())
    return state_dict_keep


def save_checkpoint(model: torch.nn.Module,
                    filename: str,
                    optimizer: Optional[Optimizer] = None,
                    meta: Optional[dict] = None,
                    file_client_args: Optional[dict] = None) -> None:
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.
            `New in version 1.3.16.`
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    if is_module_wrapper(model):
        model = model.module

    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        # save class name to the meta
        meta.update(CLASSES=model.CLASSES)

    freeze_layers = model.freeze_layers if hasattr(
        model, 'freeze_layers') else []
    checkpoint = {
        'meta': meta,
        # type: ignore
        'state_dict': keep_part_model(weights_to_cpu(get_state_dict(model)), freeze_layers)
    }
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()

    file_client = FileClient.infer_client(file_client_args, filename)
    with io.BytesIO() as f:
        torch.save(checkpoint, f)
        file_client.put(f.getvalue(), filename)
