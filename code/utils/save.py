import os
import fcntl
import shutil
import torch
from os.path import join
from utils.logger import LOGGER




class ModelSaver(object):
    def __init__(self, output_dir, prefix='model_step', suffix='pt',remove_before_ckpt=True,
                 checkpoint_lock=None, save_optimizer=True):
        self.output_dir = output_dir
        self.prefix = prefix
        self.suffix = suffix
        self.remove_before_ckpt = remove_before_ckpt
        self.checkpoint_lock = checkpoint_lock
        self.save_optimizer = save_optimizer
    def save(self, model, step, optimizer=None, best_indicator=None, save_best=False):
        if not self.save_optimizer:
            optimizer = None
        if self.checkpoint_lock:
            with open(self.checkpoint_lock, 'a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                needed_gib = 18 if optimizer is not None else 8
                if shutil.disk_usage(self.output_dir).free < needed_gib * 2**30:
                    raise RuntimeError('Insufficient space for atomic checkpoint; retaining existing files')
                return self._save_impl(model, step, optimizer, best_indicator, save_best)
        return self._save_impl(model, step, optimizer, best_indicator, save_best)

    def _save_impl(self, model, step, optimizer=None, best_indicator=None, save_best=False):
        ###remove previous model
        previous_state = [i  for i in os.listdir(self.output_dir) if i.startswith('model')]
        # if not self.pretraining:
        output_model_file = join(self.output_dir,
                                 f"{self.prefix}_{step}.{self.suffix}")
        state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in model.state_dict().items()}
        torch.save(state_dict, output_model_file + '.tmp')
        os.replace(output_model_file + '.tmp', output_model_file)

        if save_best:
            for k in best_indicator:
                if best_indicator[k]:
                    torch.save(state_dict, join(self.output_dir,
                                 f"best_{k}.{self.suffix}"))

        if optimizer is not None:
            if hasattr(optimizer, '_amp_stash'):
                pass  # TODO fp16 optimizer
            previous_state = [i  for i in os.listdir(self.output_dir) if i.startswith('optimizer')]
            optimizer_file = f'{self.output_dir}/optimizer_step_{step}.pt'
            torch.save(optimizer.state_dict(), optimizer_file + '.tmp')
            os.replace(optimizer_file + '.tmp', optimizer_file)
        # Retain the previous complete pair until the new pair is fully saved.
        if self.remove_before_ckpt:
            keep = {f'{self.prefix}_{step}.{self.suffix}', f'optimizer_step_{step}.pt'}
            for name in os.listdir(self.output_dir):
                if name.startswith(('model_step_', 'optimizer_step_')) and name.endswith('.pt') and name not in keep:
                    os.remove(os.path.join(self.output_dir, name))
