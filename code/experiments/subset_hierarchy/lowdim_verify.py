"""CPU validation, sealed readiness evidence, and local-only portable export."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.subset_hierarchy.lowdim_common import *


def evidence_root(roots):
    path=roots['logs']/'cpu_validation';path.mkdir(parents=True,exist_ok=True);return path


def record(roots,name,value):
    value.update(passed=True,source_sha256=digest(source_manifest(roots)),protocol_sha256=file_hash(roots['suite']/'protocol.json'))
    write_json(evidence_root(roots)/(name+'.json'),value)


def run_tests(roots):
    commands=[['-m','unittest','discover','-s','tests','-p','test_lowdim_*.py','-v'],
        ['tests/check_lowdim_distributed.py',*sum((['--'+key,str(roots[value])] for key,value in [('suite','suite'),('author-root','authors'),('data-root','data')]),[])]]
    outputs=[]
    for i,command in enumerate(commands):
        path=evidence_root(roots)/f'tests_{i}.log'
        with path.open('w') as log:
            code=subprocess.run([sys.executable,'-B',*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')).returncode
        if code:raise RuntimeError(f'CPU checks failed: {path}')
        outputs.append(dict(log=path.name,sha256=file_hash(path)))
    record(roots,'tests',dict(checks=outputs,scope='CPU geometry, state restoration, scheduler and distributed objective fixtures; no full CUDA encoder validation'))


def check_models(roots):
    import torch
    from experiments.subset_hierarchy.lowdim_models import make_model
    from utils.lowdim_geometry import initialize_subset_heads,project_native_state
    from experiments.subset_hierarchy.lowdim_data import CachedDataset,collate
    sample=CachedDataset(roots['suite'],'msrvtt','val')[0]
    def forward_check(net,method):
        net.eval();batch=collate([sample])
        with torch.no_grad():
            features=net(batch) if ours(method) else {m:net.batch_get(batch,'feat_'+m) for m in 'tva'}
        if any(v.shape!=(1,512) or not torch.isfinite(v).all() for v in features.values()):raise AssertionError('Cached input/encoder feature mismatch')
        return dict(id=sample['ids'],features={k:list(v.shape) for k,v in features.items()},device='cpu',backward=False)
    torch.set_num_threads(4);p=read_json(roots['suite']/'protocol.json');results={}
    for family in ('vast','gram','pmrl'):
        checkpoint=resolve_ref(p['checkpoints'][family],roots)
        if file_hash(checkpoint)!=p['checkpoint_sha256'][family]:raise ValueError(f'Checkpoint changed: {family}')
        torch.manual_seed(50)
        net,audit=make_model(roots,'activitynet',family,512,'cpu',stage_local=False)
        if net.vision_frame_embedding.shape[1]!=8 or net.audio_frame_embedding.shape[1]!=2:raise AssertionError('Frame embeddings not adapted')
        state=net.state_dict();projection_results={}
        for width in DIMENSIONS:
            projected,keys=project_native_state(state,torch.eye(512),width)
            projection_results[width]={k:list(projected[k].shape) for k in keys}
            del projected
        if family=='vast':
            for width in DIMENSIONS:
                reference=None
                for method in ('hyp_cone','hyp_none','euc_cone','euc_none'):
                    heads=initialize_subset_heads(p['input_dims'],state,torch.eye(512),width,geometry_for(method))
                    current=heads.state_dict()
                    if reference is not None:
                        for key in current:torch.testing.assert_close(current[key],reference[key],atol=0,rtol=0)
                    else:reference={k:v.clone() for k,v in current.items()}
                del heads,reference,current
        results[family]=dict(audit=audit,projected_shapes=projection_results,cached_input_forward=forward_check(net,family))
        del state,net;gc.collect();print('CPU constructor/checkpoint PASS',family,flush=True)
    torch.manual_seed(50);net,audit=make_model(roots,'msrvtt','hyp_cone',512,'cpu',stage_local=False)
    results['subset']=dict(audit=audit,cached_input_forward=forward_check(net,'hyp_cone'));del net;gc.collect()
    if torch.cuda.is_initialized():raise AssertionError('CPU audit unexpectedly initialized CUDA')
    record(roots,'models',dict(models=results,scope='Full CPU constructors/checkpoint loads, projected native head shapes and four-arm identical initialization. One cached MSRVTT sample also passes each complete CPU encoder forward; CUDA and full encoder backward remain pending.'))


def check_cache(roots):
    import numpy as np
    import torch
    from easydict import EasyDict as edict
    from experiments.subset_hierarchy.lowdim_data import CachedDataset
    from data.audio_mapper import AudioMapper
    torch.set_num_threads(1);results={}
    for name in DATASETS:
        directory=roots['suite']/'data'/name;ready=read_json(directory/'READY.json')
        if not ready['passed'] or ready['settings']!=DATA_SETTINGS[name] or ready['cache_settings']!=CACHE_SETTINGS:raise ValueError('Cache protocol mismatch')
        for split in ('train','val','test'):
            if file_hash(directory/f'{split}.json')!=ready['manifest_hashes'][split]:raise ValueError('Manifest changed')
        for filename,key in [('token_ids.npy','token_ids_sha256'),('token_mask.npy','token_mask_sha256')]:
            if file_hash(directory/filename)!=ready[key]:raise ValueError('Token cache changed')
        rows={s:read_json(directory/f'{s}.json') for s in ('train','val','test')};ids=[r['clip_id'] for split in rows.values() for r in split]
        if len(ids)!=len(set(ids)):raise AssertionError('Duplicate ID or split leakage')
        metadata={}
        for index,identifier in enumerate(ids):
            media=directory/'media'/identifier;meta=read_json(media/'meta.json')
            for filename,key,dtype in [('frames.npy','frames_shape','uint8'),('fbank.npy','fbank_shape','float32')]:
                array=np.load(media/filename,mmap_mode='r')
                if list(array.shape)!=meta[key] or array.dtype!=np.dtype(dtype):raise ValueError('Invalid cached array')
                del array
            metadata[identifier]=digest(meta)
            if index%2000==0:print('Audit cache',name,index,'/',len(ids),flush=True)
        # Sample across each frozen split; compare old AudioMapper evaluation
        # windows and ensure stateless augmentation survives global RNG changes.
        audio=resolve_ref(ready['sources']['audio'],roots)
        mapper=AudioMapper(edict(audio=str(audio),training=False,audio_sample_num=ready['settings']['audio_windows'],strict_media=True),
            edict(model_cfg=edict(audio_melbins=64,audio_target_length=1024,audio_encoder_type='beats')))
        checks=[]
        for split in ('train','val','test'):
            ds=CachedDataset(roots['suite'],name,split,training=False)
            for index in (0,len(ds)//2,len(ds)-1):
                sample=ds[index];expected=mapper.read(sample['ids'])
                torch.testing.assert_close(sample['audio_spectrograms'],expected,rtol=0,atol=0)
                if not torch.isfinite(sample['vision_pixels']).all():raise ValueError('Invalid cached pixels')
                ds.set_epoch(4);again=ds[index]
                torch.testing.assert_close(sample['vision_pixels'],again['vision_pixels'],rtol=0,atol=0)
                checks.append(sample['ids'])
        ds=CachedDataset(roots['suite'],name,'train');ds.set_epoch(1);first=ds[0]
        torch.manual_seed(123);second=ds[0]
        for key in ('vision_pixels','audio_spectrograms'):torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
        ds.set_epoch(2);third=ds[0]
        if torch.equal(first['vision_pixels'],third['vision_pixels']):raise AssertionError('Training augmentation did not vary across epochs')
        results[name]=dict(counts=ready['counts'],excluded=ready['excluded'],metadata_sha256=digest(metadata),audio_exact_match_ids=checks,
            ready_sha256=file_hash(directory/'READY.json'))
    record(roots,'cache',dict(datasets=results))


def seal(roots):
    signature=digest(source_manifest(roots));evidence={}
    for name in ('tests','models','cache'):
        path=evidence_root(roots)/(name+'.json');result=read_json(path)
        if not result['passed'] or result['source_sha256']!=signature or result['protocol_sha256']!=file_hash(roots['suite']/'protocol.json'):
            raise RuntimeError(f'{name} evidence is stale: rerun CPU validation after source changes')
        evidence[name]=result
    ready={d:file_hash(roots['suite']/'data'/d/'READY.json') for d in DATASETS}
    if any(evidence['cache']['datasets'][d]['ready_sha256']!=ready[d] for d in DATASETS):raise RuntimeError('Cache audit is stale')
    write_json(roots['suite']/'CPU_READY.json',dict(passed=True,runtime_versions=runtime_versions(),protocol_sha256=file_hash(roots['suite']/'protocol.json'),training_sources=source_manifest(roots),
        data_ready_sha256=ready,evidence=evidence,gpu_status='PENDING: no CUDA jobs launched by CPU verification'))


def export(roots):
    require_cpu_gate(roots)
    # Local export only; no network, SSH, uploads or credentials.
    target=roots['workspace']/'handoff'/('lowdim_'+str(time.time_ns()));target.mkdir(parents=True)
    files=source_files(roots)
    docs=ROOT/'experiments/subset_hierarchy/LOWDIM_README.md'
    files['code/experiments/subset_hierarchy/LOWDIM_README.md']=docs
    files['code/requirements-lowdim-finetune.txt']=ROOT/'requirements-lowdim-finetune.txt'
    for key,source in files.items():
        dest=target/key;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    for filename in ('protocol.json','tasks.json','CPU_READY.json'):shutil.copy2(roots['suite']/filename,target/filename)
    p=read_json(roots['suite']/'protocol.json');resources={}
    for name,ref in p['checkpoints'].items():resources['checkpoint_'+name]=dict(resource=ref,sha256=p['checkpoint_sha256'][name],bytes=resolve_ref(ref,roots).stat().st_size)
    assets=resolve_ref(p['assets'],roots)
    required=[assets/'clip/EVA01_CLIP_g_14_psz14_s11B.pt',assets/'beats/BEATs_iter3_plus_AS2M.pt',*sorted((assets/'bert/bert-base-uncased').rglob('*'))]
    for path in required:
        if path.is_file():resources['asset/'+str(path.relative_to(assets))]=dict(resource=path_ref(path,roots),bytes=path.stat().st_size,sha256=file_hash(path))
    resources['suite']=dict(copy_entire_directory=str(roots['suite']),contents='protocol/tasks/CPU_READY/data; preserve relative paths; raw videos are only needed to repeat preprocessing')
    versions={}
    for name in ['torch','torchvision','torchaudio','numpy','transformers','easydict','decord','Pillow','h5py','toolz','ftfy','regex','einops','imageio-ffmpeg','tqdm']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='missing'
    write_json(target/'resources.json',resources);write_json(target/'environment.json',dict(python=sys.version,packages=versions))
    write_json(target/'SHA256.json',{str(p.relative_to(target)):file_hash(p) for p in sorted(target.rglob('*')) if p.is_file()})
    archive=target.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(target,arcname=target.name)
    write_json(roots['logs']/'handoff.json',dict(directory=str(target),archive=str(archive),sha256=file_hash(archive),uploaded=False))
    print('Local portable bundle:',archive,flush=True)


def main():
    p=argparse.ArgumentParser();roots_args(p);p.add_argument('--stage',choices=['tests','models','cache','seal','export','all'],default='all')
    args=p.parse_args();roots=get_roots(args)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':p.error("CPU verification requires CUDA_VISIBLE_DEVICES='' explicitly")
    append_json(evidence_root(roots)/'launchers.jsonl',dict(pid=os.getpid(),command=sys.argv,gpu='',time=time.time()))
    stages=['tests','models','cache','seal','export'] if args.stage=='all' else [args.stage]
    for stage in stages:{'tests':run_tests,'models':check_models,'cache':check_cache,'seal':seal,'export':export}[stage](roots)

if __name__=='__main__':main()
