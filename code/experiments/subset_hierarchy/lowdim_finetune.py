"""Explicit GPU stages for the portable, paired low-dimensional experiment suite.

Nothing launches a GPU stage without --allow-gpu. CPU preparation and handoff
are intentionally separate, so copying this suite never starts training.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from experiments.subset_hierarchy.lowdim_common import *
from experiments.subset_hierarchy.lowdim_data import CachedDataset,GlobalBatchSampler,collate
from experiments.subset_hierarchy.lowdim_models import make_model,TrainingAdapter
from utils.lowdim_geometry import loss as subset_loss,retrieval_metrics,geometry_metrics,fit_basis


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def to_device(value,device):
    if torch.is_tensor(value):return value.to(device,non_blocking=True)
    if isinstance(value,dict):return edict({k:to_device(v,device) for k,v in value.items()})
    return value


def loader(ds,rank,world,epoch,training,workers=2):
    ds.set_epoch(epoch)
    sampler=GlobalBatchSampler(len(ds),rank,world,64,50,epoch,training)
    return DataLoader(ds,batch_sampler=sampler,num_workers=workers,collate_fn=collate,pin_memory=True,
        generator=torch.Generator().manual_seed(50+epoch),persistent_workers=False,
        **({'multiprocessing_context':'spawn','timeout':180} if workers else {}))


def capture_rng(device):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device))


def restore_rng(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'],device)


def gather_object(value,rank,world):
    if world==1:return [value]
    result=[None]*world if rank==0 else None
    dist.gather_object(value,result,dst=0)
    return result


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


@contextmanager
def checkpoint_slot(suite):
    handles=[(suite/f'checkpoint_writer_{i}.lock').open('a') for i in range(2)];chosen=None
    try:
        while chosen is None:
            for h in handles:
                try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);chosen=h;break
                except BlockingIOError:pass
            if chosen is None:time.sleep(.5)
        if shutil.disk_usage(suite).free<30*2**30:raise RuntimeError('Insufficient free space for atomic checkpoint')
        yield
    finally:
        for h in handles:h.close()


def save_checkpoint(suite,path,payload):
    with checkpoint_slot(suite):
        temporary=path.with_name(path.name+f'.{os.getpid()}.partial')
        torch.save(cpu_tree(payload),temporary);os.replace(temporary,path)


def point_best(run,filename):
    target=run/'best.pt';temporary=run/f'best.{os.getpid()}.partial'
    if temporary.is_symlink():temporary.unlink()
    temporary.symlink_to(filename);os.replace(temporary,target)


def write_history(run,history):
    temporary=run/'epochs.jsonl.partial'
    temporary.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in history));os.replace(temporary,run/'epochs.jsonl')


def commit_epoch(suite,run,net,optimizer,scaler,config,summary,history,rngs,best,best_epoch,best_file,improved):
    old_best=best_file
    if improved:
        best_file=f'best_epoch_{summary["epoch"]:04d}.pt'
        save_checkpoint(suite,run/best_file,dict(model=net.state_dict(),epoch=summary['epoch'],validation=summary['validation'],config=config))
    payload=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
        epoch=summary['epoch'],step=summary['step'],best=best,best_epoch=best_epoch,best_file=best_file,
        history=history,rngs=rngs,config=config)
    save_checkpoint(suite,run/'last.pt',payload)
    point_best(run,best_file);write_history(run,history);write_json(run/'status.json',summary)
    # Only prune this run's own superseded rolling-best files after committing
    # last.pt. The selected model and exact resumable last state both remain.
    if old_best and old_best!=best_file:
        path=run/old_best
        if path.parent.resolve()!=run.resolve() or not path.name.startswith('best_epoch_'):raise ValueError('Unsafe rolling checkpoint path')
        if path.exists():
            append_json(run/'checkpoint_pruning.jsonl',dict(path=str(path),superseded_by=best_file,epoch=summary['epoch']))
            path.unlink()
    return best_file


@torch.no_grad()
def extract(net,method,ds,rank,world,device,workers=2):
    net.eval();names=list('tva') if not ours(method) else ['t','v','a','tv','ta','va','tva']
    values={s:[] for s in names};ids=[]
    for batch in loader(ds,rank,world,0,False,workers):
        ids.extend(batch['ids']);batch=to_device(batch,device)
        with torch.autocast('cuda',dtype=torch.float16):
            if ours(method):z=net(batch)
            else:z={s:net.batch_get(batch,'feat_'+s) for s in names}
        for s in names:values[s].append(z[s].detach().cpu())
    local=dict(ids=ids,features={s:torch.cat(v) for s,v in values.items()})
    pieces=gather_object(local,rank,world)
    if rank!=0:return None
    all_ids=sum([p['ids'] for p in pieces],[]);index={v:i for i,v in enumerate(all_ids)}
    expected=[r['clip_id'] for r in ds.rows]
    if len(all_ids)!=len(index) or set(all_ids)!=set(expected):raise RuntimeError('Feature IDs are missing or duplicated')
    order=torch.tensor([index[v] for v in expected])
    merged={s:torch.cat([p['features'][s] for p in pieces])[order] for s in names}
    if any(not torch.isfinite(v).all() for v in merged.values()):raise FloatingPointError('Nonfinite extracted features')
    return dict(ids=expected,features=merged)


def evaluate(net,method,ds,rank,world,device,workers):
    pack=extract(net,method,ds,rank,world,device,workers)
    if rank==0:
        metric='cosine' if not ours(method) else geometry_for(method)
        result,ranks=retrieval_metrics(pack['features'],metric)
        if ours(method):result['geometry']=geometry_metrics(pack['features'],metric)
        result['n_candidates']=len(pack['ids']);pack['ranks']=ranks
    else:result=None
    value=[result];dist.broadcast_object_list(value,src=0)
    return value[0],pack


def source_signature(roots):
    return digest(source_manifest(roots))


def fit_saved_basis(roots,dataset,family):
    directory=roots['suite']/'bases'/dataset;pack=torch.load(directory/(family+'_features.pt'),map_location='cpu')
    ready=read_json(roots['suite']/'data'/dataset/'READY.json');rows=read_json(roots['suite']/'data'/dataset/'train.json')
    if pack['ids']!=[r['clip_id'] for r in rows]:raise ValueError('PCA input is not exactly the training cohort')
    p=read_json(roots['suite']/'protocol.json')
    if pack['checkpoint_sha256']!=p['checkpoint_sha256'][family] or pack['train_manifest_sha256']!=ready['manifest_hashes']['train']:
        raise ValueError('PCA feature provenance mismatch')
    basis,values=fit_basis(pack['features'])
    result=dict(basis=basis,eigenvalues=values,train_ids_sha256=digest(pack['ids']),train_manifest_sha256=ready['manifest_hashes']['train'],
        checkpoint_sha256=p['checkpoint_sha256'][family],cache_settings_sha256=digest(ready['cache_settings']))
    temporary=directory/(family+'.partial');torch.save(result,temporary);os.replace(temporary,directory/(family+'.pt'))
    write_json(directory/(family+'_READY.json'),dict(passed=True,n_train=len(pack['ids']),sha256=file_hash(directory/(family+'.pt'))))


def worker(rank,args,rendezvous):
    roots=get_roots(args);world=args.world_size;suite=roots['suite'];device=torch.device('cuda',rank)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);seed_all(50);torch.cuda.set_device(device)
    free,total=torch.cuda.mem_get_info(device);cap=args.memory_cap_gib*2**30 if args.memory_cap_gib else free-3*2**30
    if cap<6*2**30 or cap>free-3*2**30:raise RuntimeError('GPU headroom insufficient for requested allocation cap')
    torch.cuda.set_per_process_memory_fraction(cap/total,device)
    dist.init_process_group('nccl',init_method='file://'+rendezvous,rank=rank,world_size=world)
    stage=args.stage;name=trial_name(args.dataset,args.method,args.dimension)
    parent='runs' if stage=='train' else ('bases' if stage=='basis' else 'profiles')
    run=suite/parent/(args.dataset if stage=='basis' else name)
    if stage not in ('train','basis'):run=run/(f'{stage}_w{world}_'+args.tag)
    if rank==0:run.mkdir(parents=True,exist_ok=True)
    dist.barrier()
    append_json(roots['logs']/name/f'{stage}_launchers.jsonl',dict(pid=os.getpid(),rank=rank,world=world,gpu_visible=os.environ['CUDA_VISIBLE_DEVICES'],
        device=rank,command=sys.argv,stage=stage,tag=args.tag,run=str(run)))
    net,audit=make_model(roots,args.dataset,args.method,512 if stage=='basis' else args.dimension,device,initial_only=stage=='basis')
    if rank==0:write_json(run/('init_audit_'+args.method+'.json' if stage=='basis' else 'init_audit.json'),audit)
    if stage=='basis':
        if args.method not in ('vast','gram','pmrl'):raise ValueError('PCA extraction uses a pretrained baseline family')
        ds=CachedDataset(suite,args.dataset,'train',training=False)
        pack=extract(net,args.method,ds,rank,world,device,args.workers)
        if rank==0:
            p=read_json(suite/'protocol.json');ready=read_json(suite/'data'/args.dataset/'READY.json')
            pack.update(checkpoint_sha256=p['checkpoint_sha256'][args.method],train_manifest_sha256=ready['manifest_hashes']['train'])
            temp=run/(args.method+'_features.partial');torch.save(pack,temp);os.replace(temp,run/(args.method+'_features.pt'))
            fit_saved_basis(roots,args.dataset,args.method)
        dist.barrier();dist.destroy_process_group();return
    protocol=read_json(suite/'protocol.json');ready=read_json(suite/'data'/args.dataset/'READY.json')
    config=dict(dataset=args.dataset,method=args.method,dimension=args.dimension,seed=50,world_size=world,
        protocol_sha256=file_hash(suite/'protocol.json'),data=ready,source=source_signature(roots),initialization=audit,
        precision='fp16_encoders_fp64_geometry',amp_initial_scale=args.amp_scale)
    adapter=TrainingAdapter(net,args.method)
    train_model=DDP(adapter,device_ids=[rank],find_unused_parameters=True,broadcast_buffers=False) if world>1 else adapter
    groups=[]
    for visual in (True,False):
        params=[v for k,v in net.named_parameters() if v.requires_grad and k.startswith('vision_encoder.visual.')==visual]
        lr=protocol['vision_learning_rate'] if visual else protocol['learning_rate']
        groups.append(dict(params=params,lr=lr,initial_lr=lr))
    optimizer=torch.optim.AdamW(groups,betas=tuple(protocol['betas']),weight_decay=protocol['weight_decay'],foreach=False)
    scaler=torch.cuda.amp.GradScaler(init_scale=args.amp_scale)
    train=CachedDataset(suite,args.dataset,'train',limit=128 if stage!='train' else 0)
    val=CachedDataset(suite,args.dataset,'val',limit=32 if stage!='train' else 0)
    seed_all(50+rank);step=0;start_epoch=1;best=-1.;best_epoch=0;best_file=None;history=[]
    if stage=='train' and args.resume and (run/'last.pt').exists():
        saved=torch.load(run/'last.pt',map_location='cpu')
        if saved['config']!=config:raise ValueError('Resume protocol/source/world-size differs')
        net.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        step=saved['step'];start_epoch=saved['epoch']+1;best=saved['best'];best_epoch=saved['best_epoch'];best_file=saved['best_file'];history=saved['history']
        restore_rng(saved['rngs'][rank],device);del saved
        if rank==0:write_history(run,history);point_best(run,best_file)
    elif stage=='train':
        if (run/'last.pt').exists():raise RuntimeError('Existing training state requires --resume')
        validation,_=evaluate(net,args.method,val,rank,world,device,args.workers)
        best=validation['macro']['R@1'];best_file='best_epoch_0000.pt'
        if rank==0:
            write_json(run/'initial_validation.json',validation);write_json(run/'config.json',config)
            save_checkpoint(suite,run/best_file,dict(model=net.state_dict(),epoch=0,validation=validation,config=config));point_best(run,best_file)
        dist.barrier()
    steps_epoch=len(train)//64;total_steps=(len(CachedDataset(suite,args.dataset,'train'))//64)*10
    warmup=max(1,int(total_steps*.05));timings=[];profile_updates=0
    epochs=range(start_epoch,11) if stage=='train' else range(1,math.ceil(args.steps/max(steps_epoch,1))+1)
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in epochs:
        train_model.train();started=time.monotonic();totals={};updates=0
        iterator=iter(loader(train,rank,world,epoch,True,args.workers));batch_index=0
        while True:
            tick=time.monotonic()
            try:batch=next(iterator)
            except StopIteration:break
            batch_index+=1;batch=to_device(batch,device);optimizer.zero_grad(set_to_none=True)
            factor=(step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
            for group in optimizer.param_groups:group['lr']=group['initial_lr']*factor
            with torch.autocast('cuda',dtype=torch.float16):output=train_model(batch)
            if ours(args.method):
                value,terms=subset_loss(output,geometry_for(args.method),.1 if args.method.endswith('_cone') else 0.)
                if int(terms['candidate_count'])!=64:raise RuntimeError('Incorrect contrastive candidate pool')
            else:terms=output;value=sum(output.values())
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite training loss')
            scaler.scale(value).backward();scaler.unscale_(optimizer)
            grad_norm=torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],2.)
            bad=torch.tensor(int(not torch.isfinite(grad_norm)),device=device);dist.all_reduce(bad,op=dist.ReduceOp.MAX)
            if bad.item():raise FloatingPointError('Nonfinite gradient: failed run, no silently skipped update')
            branches={}
            if batch_index==1:
                prefixes=['vision_encoder.visual.','audio_encoder.','multimodal_encoder.bert.']
                prefixes+=([f'subset_heads.direction.{s}.' for s in ('t','v','a','tv','ta','va','tva')] if ours(args.method) else [f'contra_head_{s}.' for s in 'tva'])
                for prefix in prefixes:
                    key,param=next((k,p) for k,p in net.named_parameters() if k.startswith(prefix) and k.endswith('weight') and p.grad is not None)
                    branches[key]=float(param.grad.norm())
                if not all(math.isfinite(v) and v>0 for v in branches.values()):raise FloatingPointError('Missing encoder/head gradient')
            scaler.step(optimizer);scaler.update();step+=1;updates+=1;profile_updates+=1
            torch.cuda.synchronize(device);duration=time.monotonic()-tick;timings.append(duration)
            for key,value_ in terms.items():
                if key!='candidate_count':totals[key]=totals.get(key,0.)+float(value_.detach())
            if rank==0 and (batch_index==1 or batch_index%25==0):
                record=dict(epoch=epoch,step=step,batch=batch_index,loss=float(value.detach()),grad_norm=float(grad_norm),
                    seconds_per_batch=(time.monotonic()-started)/batch_index,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,branches=branches)
                append_json(run/'steps.jsonl',record);print(json.dumps(record),flush=True)
            del batch,output,value
            if stage!='train' and profile_updates>=args.steps:break
        if stage!='train':
            if profile_updates>=args.steps:break
            continue
        validation,_=evaluate(net,args.method,val,rank,world,device,args.workers)
        metric=validation['macro']['R@1'];improved=metric>best+1e-10
        if improved:best=metric;best_epoch=epoch
        keys=sorted(totals);numbers=torch.tensor([totals[k]/updates for k in keys],dtype=torch.float64,device=device)
        dist.all_reduce(numbers);numbers/=world
        summary=dict(epoch=epoch,step=step,updates=updates,skipped=0,losses=dict(zip(keys,numbers.tolist())),validation=validation,
            best=best,best_epoch=best_epoch,elapsed=time.monotonic()-started)
        history.append(summary);rngs=gather_object(capture_rng(device),rank,world)
        if rank==0:
            best_file=commit_epoch(suite,run,net,optimizer,scaler,config,summary,history,rngs,best,best_epoch,best_file,improved)
            print(json.dumps(summary),flush=True)
        objects=[best_file];dist.broadcast_object_list(objects,src=0);best_file=objects[0];dist.barrier()
    if stage!='train':
        validation,_=evaluate(net,args.method,val,rank,world,device,args.workers)
        local=dict(peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
            durations=timings[2:] if len(timings)>2 else timings,rank=rank)
        profiles=gather_object(local,rank,world)
        if rank==0:
            seconds=max(sum(p['durations'])/len(p['durations']) for p in profiles)
            report=dict(passed=True,stage=stage,world_size=world,global_batch=64,steps=profile_updates,seconds_per_step=seconds,
                samples_per_second=64/seconds,ranks=profiles,validation=validation,config=config,tag=args.tag)
            write_json(run/'DONE.json',report)
    else:
        del optimizer,scaler
        saved=torch.load(run/best_file,map_location='cpu');net.load_state_dict(saved['model'],strict=True);selected=saved['epoch'];del saved
        test=CachedDataset(suite,args.dataset,'test');metrics,features=evaluate(net,args.method,test,rank,world,device,args.workers)
        if rank==0:
            metrics.update(selected_epoch=selected,config=config,epochs_completed=10,steps=step)
            write_json(run/'test_metrics.json',metrics);torch.save(features,run/'test_features.pt')
            write_json(run/'DONE.json',dict(passed=True,selected_epoch=selected,epochs=10,steps=step,macro_R1=metrics['macro']['R@1'],test_ids_sha256=digest(features['ids'])))
    dist.barrier();dist.destroy_process_group()


def validate_completed(roots,run):
    done=read_json(run/'DONE.json');metrics=read_json(run/'test_metrics.json');cfg=metrics['config']
    if not done.get('passed') or done['epochs']!=10 or cfg['source']!=source_signature(roots) or cfg['protocol_sha256']!=file_hash(roots['suite']/'protocol.json'):
        raise RuntimeError('Existing completed run has stale or incomplete provenance')
    if cfg['data']!=read_json(roots['suite']/'data'/cfg['dataset']/'READY.json'):
        raise RuntimeError('Existing completed run used a different data cohort')


def main():
    p=argparse.ArgumentParser();roots_args(p)
    p.add_argument('--stage',choices=['inspect','fit-basis','basis','smoke','profile','train'],default='inspect')
    p.add_argument('--dataset',choices=DATASETS,default='msrvtt');p.add_argument('--method',choices=METHODS,default='vast')
    p.add_argument('--dimension',type=int,choices=DIMENSIONS,default=512);p.add_argument('--devices',default='')
    p.add_argument('--seed',type=int,choices=[50],default=50)
    p.add_argument('--world-size',type=int,choices=[1,2,4,8],default=1);p.add_argument('--allow-gpu',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--steps',type=int,default=12);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--memory-cap-gib',type=float,default=0.);p.add_argument('--amp-scale',type=float,default=1024.)
    p.add_argument('--tag',default='default');args=p.parse_args();roots=get_roots(args)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.tag):p.error('Unsafe run tag')
    if not 0<=args.workers<=4:p.error('Worker count must be between zero and four')
    if args.stage=='inspect':
        protocol=read_json(roots['suite']/'protocol.json')
        print(json.dumps(dict(tasks=len(read_json(roots['suite']/'tasks.json')),gpu_launch_enabled=False,
            datasets={d:(read_json(roots['suite']/'data'/d/'READY.json')['counts'] if (roots['suite']/'data'/d/'READY.json').exists() else 'CPU cache pending') for d in DATASETS},
            bases={d:{n:(roots['suite']/'bases'/d/(n+'_READY.json')).exists() for n in ['vast','gram','pmrl']} for d in DATASETS}),indent=2));return
    if args.stage=='fit-basis':fit_saved_basis(roots,args.dataset,args.method);return
    if not args.allow_gpu:p.error('GPU stages require explicit --allow-gpu; this suite never starts them automatically')
    if not (roots['suite']/'CPU_READY.json').exists():p.error('CPU verification and handoff are not complete')
    require_cpu_gate(roots)
    devices=[v.strip() for v in args.devices.split(',') if v.strip()]
    if len(devices)!=args.world_size or len(set(devices))!=len(devices):p.error('--devices must contain exactly world-size distinct GPU indices')
    if args.stage=='train' and not (roots['suite']/'GPU_READY.json').exists():p.error('GPU smoke/distributed validation gate is pending')
    if args.stage=='train':
        require_gpu_gate(roots)
        from experiments.subset_hierarchy.lowdim_schedule import load_layouts
        load_layouts(roots,devices)
    if args.steps<2:p.error('At least two steps are required to allocate optimizer state')
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(devices)
    name=trial_name(args.dataset,args.method,args.dimension);log=roots['logs']/name;log.mkdir(parents=True,exist_ok=True)
    rendezvous=str(log/f'dist_{os.getpid()}_{time.time_ns()}')
    lock_dir=roots['suite']/('runs' if args.stage=='train' else 'stage_locks')/name
    lock_dir.mkdir(parents=True,exist_ok=True)
    try:
        lock_name=args.stage if args.stage in ('train','basis') else f'{args.stage}_w{args.world_size}_{args.tag}'
        with (lock_dir/(lock_name+'.lock')).open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.stage=='train' and (lock_dir/'DONE.json').exists():
                validate_completed(roots,lock_dir)
                print('Already completed:',name);return
            if args.world_size==1:worker(0,args,rendezvous)
            else:mp.spawn(worker,args=(args,rendezvous),nprocs=args.world_size,join=True)
    except Exception:
        write_json(log/f'{args.stage}_failure_{time.time_ns()}.json',dict(command=sys.argv,error=traceback.format_exc()))
        raise

if __name__=='__main__':main()
