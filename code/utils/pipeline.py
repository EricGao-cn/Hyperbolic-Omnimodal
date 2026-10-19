import os
import json
import math
import time
import torch 
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm 
# from apex import amp
from collections import defaultdict
from torch.nn.utils import clip_grad_norm_
from evaluation import evaluation_registry
from .save import ModelSaver
from .tool import NoOp
from .logger import LOGGER, RunningMeter
from .sched import get_lr_sched
from torch.cuda.amp import autocast, GradScaler
import wandb



def train(model, optimizer, train_loader, val_loaders, run_cfg, start_step=0, verbose_time=False):
    stop_after = int(getattr(run_cfg, 'stop_after_updates', 0) or run_cfg.num_train_steps)
    if not start_step < stop_after <= run_cfg.num_train_steps:
        raise ValueError('stop_after_updates must exceed start_step and not exceed the scheduler horizon')
  
    if dist.get_rank() == 0:
        pbar = tqdm(total=stop_after, initial=start_step)
        model_saver = ModelSaver(os.path.join(run_cfg.output_dir, 'ckpt'),remove_before_ckpt=run_cfg.remove_before_ckpt,
                                 checkpoint_lock=getattr(run_cfg, 'checkpoint_lock', None),
                                 save_optimizer=getattr(run_cfg, 'save_optimizer', True))
    else:
        pbar = NoOp()
        model_saver = NoOp()
        
    loss_moving_averagetors ={}
    metric_logger_dict = defaultdict(dict)
    global_step = start_step

    scaler = GradScaler(init_scale=getattr(run_cfg, 'amp_init_scale', 65536.0))
    record_resources(run_cfg)
    started_at = time.monotonic()
    consecutive_skips = 0
    accumulation_steps = run_cfg.gradient_accumulation_steps
    optimizer.zero_grad()

    best_indicator = {}
    evaluate_fn = evaluation_registry[model.config.evaluation_type]

    for step, (name, batch) in enumerate(train_loader):
      
        ndata = train_loader.ndata
        task = name.split('--')[0]



        if run_cfg.fp16:
            with autocast():
                loss_dict = model(batch, task=task, compute_loss=True)
                loss = sum(list(loss_dict.values()))
                loss_dict['total_loss'] = loss
                loss_dict = {k:v.item() for k,v in loss_dict.items()}
                if dist.get_rank() == 0:
                    wandb.log(loss_dict)
                
        else:
            loss_dict = model(batch, task=task, compute_loss=True)
            loss = sum(list(loss_dict.values()))
            loss_dict['total_loss'] = loss
            loss_dict = {k:v.item() for k,v in loss_dict.items()}
            if dist.get_rank() == 0:
                    wandb.log(loss_dict)
            





        


        if not all(math.isfinite(value) for value in loss_dict.values()):
            raise FloatingPointError(f'Nonfinite loss at step {global_step}: {loss_dict}')

        if not name in loss_moving_averagetors:
                ### first time initialize 
            for k in loss_dict.keys():
                loss_moving_averagetors[f'loss_{name}/{k}'] = RunningMeter()
        ####accumulate loss

        for k,v in loss_dict.items():
            loss_moving_averagetors[f'loss_{name}/{k}'](v)
    

        if run_cfg.fp16:
            scaler.scale(loss / accumulation_steps).backward()
        else:
            (loss / accumulation_steps).backward()

        if (step + 1) % accumulation_steps != 0:
            continue

        global_step += 1
        # learning rate scheduling
        lr_ratio = get_lr_sched(global_step, run_cfg)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['init_lr'] * lr_ratio
        
        if global_step % getattr(run_cfg, 'log_steps', 50) == 0:
            LOGGER.info({name : averagetor.val for name, averagetor in loss_moving_averagetors.items()})                                   
        
        if not run_cfg.use_ddp:
            works = []
            for p in model.parameters():
            # to speed it up, you can also organize grads to larger buckets to make allreduce more efficient
                if p.grad is not None:
                    works.append(dist.all_reduce(p.grad, async_op=True))
            for work in works:
                work.wait()


        grad_norm = getattr(run_cfg, 'grad_norm', -1)
        measured_grad_norm = None
        if grad_norm != -1:
            if run_cfg.fp16:
                scaler.unscale_(optimizer)
            measured_grad_norm = float(clip_grad_norm_(model.parameters(), grad_norm))

        scale_before = scaler.get_scale()
        if run_cfg.fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        step_applied = not run_cfg.fp16 or scaler.get_scale() >= scale_before
        if not step_applied:
            global_step -= 1
            consecutive_skips += 1
        else:
            consecutive_skips = 0
        if dist.get_rank() == 0:
            record = {
                'step': global_step, 'attempt': step + 1, 'losses': loss_dict,
                'grad_norm': measured_grad_norm if measured_grad_norm is None or math.isfinite(measured_grad_norm) else None,
                'amp_scale': scaler.get_scale(),
                'optimizer_step_applied': step_applied,
                'elapsed_seconds': time.monotonic() - started_at,
                'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
            }
            if hasattr(model, 'training_diagnostics'):
                record['geometry'] = model.training_diagnostics()
            with open(os.path.join(run_cfg.output_dir, 'metrics.jsonl'), 'a') as stream:
                stream.write(json.dumps(record) + '\n')
        optimizer.zero_grad()
        if global_step % getattr(run_cfg, 'log_steps', 50) == 0:
            record_resources(run_cfg)
        if consecutive_skips >= 20:
            raise FloatingPointError('AMP skipped 20 consecutive optimizer updates')
        if not step_applied:
            continue
        pbar.update(1)


        
        if (not getattr(run_cfg, 'final_checkpoint_only', False)
                and run_cfg.valid_steps > 0 and (global_step+1) % run_cfg.valid_steps == 0):
            eval_log = evaluate_fn(model, val_loaders, run_cfg, global_step)

            if dist.get_rank() == 0:
                for task_name, val_log in eval_log.items():
                    for eval_name, metric in val_log.items():
                        eval_name = task_name +'_' +eval_name 
                        metric_logger_dict[eval_name][str(global_step)] = metric
                        LOGGER.info(f"====-evaluation--{eval_name}=====step {global_step}--===========\n")
                        LOGGER.info(metric)
                        best_name = get_best_name(eval_name, metric)
                        if best_name is not None:
                            if ('best_step' not in metric_logger_dict[eval_name]) or \
                                    (metric[best_name] >= metric_logger_dict[eval_name]['best_value']):
                                metric_logger_dict[eval_name]['best_step'] = global_step
                                metric_logger_dict[eval_name]['best_value'] = metric[best_name]
                                best_indicator[eval_name] = True 
                            else:
                                best_indicator[eval_name] = False 
                            best_step = metric_logger_dict[eval_name]['best_step']
                            LOGGER.info(f"======evaluation--{eval_name}====history best step: {best_step}=======\n")
                            LOGGER.info(metric_logger_dict[eval_name][str(best_step)])          
                
                model_saver.save(model, global_step, optimizer,best_indicator, run_cfg.save_best)
    

        if global_step >= stop_after:
            break
    if dist.get_rank() == 0:
        model_saver.save(model, global_step, optimizer, best_indicator, False)
    record_resources(run_cfg)
    pbar.close()






    
  
def test(model, test_loader, run_cfg):
    record_resources(run_cfg)
    
    evaluate_fn = evaluation_registry[model.config.evaluation_type]
    eval_log = evaluate_fn(model, test_loader, run_cfg, global_step=0)
    record_resources(run_cfg)
    if dist.get_rank()==0:  
        metrics_path = os.path.join(run_cfg.output_dir, 'metrics.json')
        with open(metrics_path + '.tmp', 'w') as stream:
            json.dump(eval_log, stream, indent=2, default=float)
        os.replace(metrics_path + '.tmp', metrics_path)
        for task_name, val_log in eval_log.items():
            for eval_name, metric in val_log.items():
                eval_name = task_name +'_' +eval_name 
                # TB_LOGGER.log_scaler_dict({f"eval/{eval_name}/test_{k}": v
                #                 for k, v in metric.items() if not isinstance(v,str)})
                LOGGER.info(f"==== evaluation--{eval_name}========\n")
                LOGGER.info(metric)




def record_resources(run_cfg):
    if not getattr(run_cfg, 'resource_profile', False):
        return
    rank = dist.get_rank()
    path = os.path.join(run_cfg.output_dir, f'rank_{rank}_resources.json')
    value = {'pid': os.getpid(), 'rank': rank, 'updated_at': time.time(),
             'master_port': os.environ.get('MASTER_PORT'),
             'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
             'reserved_gib': torch.cuda.memory_reserved() / 2**30,
             'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
    with open(path + '.tmp', 'w') as stream:
        json.dump(value, stream)
    os.replace(path + '.tmp', path)


def get_best_name(eval_name, metric):
    if eval_name.startswith('cap'):
        return 'CIDEr'
    elif eval_name.startswith('qa'):
        return 'accuracy'
    elif eval_name.startswith('ret'):
        is_audio_pairwise_stage1 = (
            ('clothov2_ret' in eval_name or 'audiocaps_ret' in eval_name)
            and '_ret_itc_ta' in eval_name
        )
        if is_audio_pairwise_stage1 and 'mean_recall' in metric:
            return 'mean_recall'
        if 'video_r1' in metric:
            return 'video_r1'
    elif eval_name.startswith('pt'):
        return None 
    else:
        raise NotImplementedError
