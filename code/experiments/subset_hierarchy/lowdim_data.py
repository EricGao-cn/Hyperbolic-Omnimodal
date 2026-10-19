"""CPU preparation and stateless shared input cache for low-dimensional fine-tuning."""
import argparse
import atexit
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import re
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,Sampler
from torchvision.transforms import functional as VF, InterpolationMode
from easydict import EasyDict as edict
from experiments.subset_hierarchy.lowdim_common import *

_READER=None

def atomic_numpy(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.partial')
    with tmp.open('wb') as f:np.save(f,value,allow_pickle=False)
    os.replace(tmp,path)


def safe_id(identifier):
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+',identifier) or identifier in ('.','..'):
        raise ValueError(f'Unsafe identifier {identifier!r}')
    return identifier


def media_signature(video,audio):
    return dict(video=[video.stat().st_size,video.stat().st_mtime_ns],
                audio=[audio.stat().st_size,audio.stat().st_mtime_ns],cache=digest(CACHE_SETTINGS))


def prepare(roots):
    suite=roots['suite'];suite.mkdir(parents=True,exist_ok=True);roots['logs'].mkdir(parents=True,exist_ok=True)
    old=read_json(roots['workspace']/'outputs/msrvtt_finetune_20261017/protocol.json')
    backbone=read_json(roots['workspace']/'outputs/msrvtt_finetune_20261017/backbone_protocol.json')
    p=protocol_base();p['model_cfg']=backbone['model_cfg'];p['input_dims']=backbone['input_dims']
    p['checkpoints']={n:path_ref(old['initial_checkpoints'][n],roots) for n in ('vast','gram','pmrl')}
    p['assets']=path_ref(Path(backbone['asset_workdir'])/'pretrained_weights',roots)
    p['gram_hps']=read_json(roots['workspace']/'outputs/author_unimodal_20261016/checkpoints/gram_hps.json')
    p['author_sources']={n:path_ref(roots['authors']/n.upper(),roots) for n in ('vast','gram')}
    for n in ('vast','gram','pmrl'):
        p.setdefault('checkpoint_sha256',{})[n]=file_hash(resolve_ref(p['checkpoints'][n],roots))
    if (suite/'protocol.json').exists():
        if read_json(suite/'protocol.json')!=p:raise ValueError('Existing protocol differs; choose a new suite')
    else:write_json(suite/'protocol.json',p)
    tasks=[dict(name=trial_name(d,m,w),dataset=d,method=m,dimension=w,seed=50) for d in DATASETS for m in METHODS for w in DIMENSIONS]
    write_json(suite/'tasks.json',tasks)
    base=roots['data']/'PMRL'
    for name in DATASETS:
        dest=suite/'data'/name;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'candidates.json').exists():continue
        video=base/'datasets'/('MSRVTT/videos/videos' if name=='msrvtt' else 'activitynet/videos')
        audio=base/'main_result_assets_20260824/decoded_wav'/name
        refs=dict(video=path_ref(video,roots),audio=path_ref(audio,roots))
        if name=='msrvtt':
            manifests=roots['workspace']/'outputs/msrvtt_finetune_20261017/manifests'
            candidates={s:read_json(manifests/f'{s}.json') for s in ('train','val','test')}
            candidates={s:[dict(clip_id=r['clip_id'],captions=r['desc'] if isinstance(r['desc'],list) else [r['desc']]) for r in rows] for s,rows in candidates.items()}
        else:
            vv={Path(e.name).stem for e in os.scandir(video) if e.name.endswith('.mp4')}
            aa={Path(e.name).stem for e in os.scandir(audio) if e.name.endswith('.wav')}
            ann=base/'datasets/annotations/activitynet'
            train=read_json(ann/'descs_ret_train.json');test=read_json(ann/'descs_ret_test.json')
            assert not {r['video_id'] for r in train}&{r['video_id'] for r in test}
            good=sorted([r for r in train if r['video_id'] in vv&aa],key=lambda r:__import__('hashlib').sha256(('activitynet50:'+r['video_id']).encode()).digest())
            convert=lambda rows:[dict(clip_id=r['video_id'],captions=[r['desc']]) for r in sorted(rows,key=lambda r:r['video_id'])]
            candidates=dict(train=convert(good[500:]),val=convert(good[:500]),test=convert([r for r in test if r['video_id'] in vv&aa]))
        for split,rows in candidates.items():
            for row in rows:safe_id(row['clip_id'])
            assert len(rows)==len({r['clip_id'] for r in rows})
        sets=[{r['clip_id'] for r in candidates[s]} for s in ('train','val','test')]
        assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
        write_json(dest/'sources.json',refs);write_json(dest/'candidates.json',candidates)
        print(name,{s:len(rows) for s,rows in candidates.items()},flush=True)
    print('Prepared 56 CPU-only task specifications.',flush=True)


def cache_one(task):
    global _READER
    torch.set_num_threads(1)
    name,identifier,vpath,apath,out=task;vpath=Path(vpath);apath=Path(apath);out=Path(out)
    try:
        signature=media_signature(vpath,apath)
        if (out/'meta.json').exists():
            meta=read_json(out/'meta.json')
            if meta.get('signature')==signature:
                frames=np.load(out/'frames.npy',mmap_mode='r');bank=np.load(out/'fbank.npy',mmap_mode='r')
                if list(frames.shape)==meta['frames_shape'] and list(bank.shape)==meta['fbank_shape']:
                    return dict(dataset=name,clip_id=identifier,passed=True,cached=True)
        from data.bounded_video import BoundedVideoReader
        if _READER is None:
            _READER=BoundedVideoReader(timeout=30);atexit.register(_READER.close)
            from multiprocessing.util import Finalize
            Finalize(_READER,_READER.close,exitpriority=10)
        frames=_READER.read(vpath,lambda length,fps:[min(length-1,int((i+.5)*length/32)) for i in range(32)])
        height,width=frames.shape[1:3];scale=256/min(height,width)
        size=(round(width*scale),round(height*scale))
        frames=np.stack([np.asarray(Image.fromarray(frame).resize(size,Image.Resampling.BILINEAR)) for frame in frames])
        from data.audio_mapper import load_audio
        import torchaudio
        waveform,sr=load_audio(apath)
        if sr!=16000:waveform=torchaudio.transforms.Resample(sr,16000)(waveform)
        if waveform.ndim!=2 or waveform.shape[0]<1:raise ValueError('Invalid decoded waveform shape')
        if not torch.isfinite(waveform).all() or waveform.shape[1]==0:raise ValueError('Invalid waveform')
        # Match the existing AudioMapper/Kaldi convention: use channel zero,
        # including for the stereo WAVs present in the local media cache.
        bank=torchaudio.compliance.kaldi.fbank(waveform*2**15,num_mel_bins=64,sample_frequency=16000,frame_length=25,frame_shift=10,channel=0)
        bank=(bank-15.41663)/(6.55582*2)
        if not torch.isfinite(bank).all() or len(bank)==0:raise ValueError('Invalid fbank')
        out.mkdir(parents=True,exist_ok=True)
        atomic_numpy(out/'frames.npy',frames.astype(np.uint8));atomic_numpy(out/'fbank.npy',bank.numpy().astype(np.float32))
        write_json(out/'meta.json',dict(signature=signature,frames_shape=list(frames.shape),fbank_shape=list(bank.shape),
            audio_seconds=waveform.shape[1]/16000,source_audio_channels=waveform.shape[0],audio_channel_used=0,
            source_height=height,source_width=width))
        return dict(dataset=name,clip_id=identifier,passed=True,cached=False)
    except Exception as exc:return dict(dataset=name,clip_id=identifier,passed=False,error=repr(exc))


def tokenize_and_finalize(roots,name,results):
    dest=roots['suite']/'data'/name;candidates=read_json(dest/'candidates.json')
    good={r['clip_id'] for r in results if r['passed']};bad=[r for r in results if not r['passed']]
    filtered={s:[r for r in rows if r['clip_id'] in good] for s,rows in candidates.items()}
    if len(filtered['train'])<64 or not filtered['val'] or not filtered['test']:raise RuntimeError('Insufficient valid data')
    from transformers import BertTokenizer
    p=read_json(roots['suite']/'protocol.json')
    tok=BertTokenizer.from_pretrained(str(resolve_ref(p['assets'],roots)/'bert/bert-base-uncased'),local_files_only=True)
    texts=[]
    for split in ('train','val','test'):
        for row in filtered[split]:
            row['token_rows']=list(range(len(texts),len(texts)+len(row['captions'])));texts.extend(row['captions'])
    ids=[];att=[];limit=DATA_SETTINGS[name]['max_caption_len']
    for start in range(0,len(texts),256):
        t=tok(texts[start:start+256],padding='max_length',truncation=True,max_length=limit,return_tensors='np')
        ids.append(t['input_ids'].astype(np.int32));att.append(t['attention_mask'].astype(np.uint8))
    atomic_numpy(dest/'token_ids.npy',np.concatenate(ids));atomic_numpy(dest/'token_mask.npy',np.concatenate(att))
    for split,rows in filtered.items():write_json(dest/f'{split}.json',rows)
    write_json(dest/'excluded.json',bad)
    ready=dict(passed=True,settings=DATA_SETTINGS[name],cache_settings=CACHE_SETTINGS,counts={s:len(v) for s,v in filtered.items()},
        candidate_counts={s:len(v) for s,v in candidates.items()},excluded=len(bad),manifest_hashes={s:file_hash(dest/f'{s}.json') for s in filtered},
        token_ids_sha256=file_hash(dest/'token_ids.npy'),token_mask_sha256=file_hash(dest/'token_mask.npy'),
        sources=read_json(dest/'sources.json'))
    write_json(dest/'READY.json',ready);print('READY',name,ready['counts'],'excluded',len(bad),flush=True)


def build_cache(roots,names,workers,limit=0):
    if not 1<=workers<=16:raise ValueError('Cache workers must be in [1,16]')
    log=roots['logs']/'cache';log.mkdir(parents=True,exist_ok=True)
    write_json(log/'launcher.json',dict(pid=os.getpid(),command=sys.argv,gpu='',workers=workers,log=str(log/'progress.jsonl')))
    for name in names:
        dest=roots['suite']/'data'/name
        if (dest/'READY.json').exists() and not limit:
            print('Already ready:',name,flush=True);continue
        candidates=read_json(dest/'candidates.json');refs=read_json(dest/'sources.json')
        video=resolve_ref(refs['video'],roots);audio=resolve_ref(refs['audio'],roots)
        identifiers=sorted({r['clip_id'] for rows in candidates.values() for r in rows})
        if limit:identifiers=identifiers[:limit]
        tasks=[(name,i,str(video/(i+'.mp4')),str(audio/(i+'.wav')),str(dest/'media'/i)) for i in identifiers]
        results=[];started=time.monotonic()
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(cache_one,t) for t in tasks]
            for future in as_completed(futures):
                result=future.result();results.append(result)
                if not result['passed']:append_json(log/'failures.jsonl',result)
                if len(results)%100==0 or len(results)==len(tasks):
                    status=dict(dataset=name,finished=len(results),total=len(tasks),failed=sum(not r['passed'] for r in results),seconds=time.monotonic()-started)
                    append_json(log/'progress.jsonl',status);print(json.dumps(status),flush=True)
        write_json(dest/('smoke_cache_results.json' if limit else 'cache_results.json'),results)
        failed=sum(not r['passed'] for r in results)
        if failed>(0 if limit else max(10,len(results)//100)):
            raise RuntimeError(f'{name}: {failed}/{len(results)} cache failures require investigation before freezing data')
        if not limit:tokenize_and_finalize(roots,name,results)


class CachedDataset(Dataset):
    def __init__(self,suite,name,split,limit=0,training=None):
        self.root=Path(suite)/'data'/name;self.name=name;self.split=split;self.epoch=0
        self.training=(split=='train') if training is None else training
        self.ready=read_json(self.root/'READY.json');self.settings=self.ready['settings']
        self.rows=read_json(self.root/f'{split}.json')
        if limit:self.rows=self.rows[:limit]
        self.tokens=np.load(self.root/'token_ids.npy',mmap_mode='r');self.masks=np.load(self.root/'token_mask.npy',mmap_mode='r')
    def __len__(self):return len(self.rows)
    def set_epoch(self,epoch):self.epoch=epoch
    def __getitem__(self,index):
        row=self.rows[index];identifier=row['clip_id'];training=self.training
        rng=random.Random(int(digest([self.name,self.epoch if training else 0,identifier,50])[:16],16))
        caption_index=rng.randrange(len(row['captions'])) if training else 0
        directory=self.root/'media'/identifier
        frames=np.load(directory/'frames.npy',mmap_mode='r');count=self.settings['vision_frames']
        parts=np.array_split(np.arange(len(frames)),count)
        indices=[int(rng.choice(part)) if training else int(part[(len(part)-1)//2]) for part in parts]
        pixels=torch.from_numpy(np.array(frames[indices],copy=True).transpose(0,3,1,2)).float()/255
        h,w=pixels.shape[-2:]
        if training:
            crop=None
            for _ in range(10):
                side=round(math.sqrt(h*w*rng.uniform(.8,1.)))
                if 0<side<=min(h,w):crop=(rng.randint(0,h-side),rng.randint(0,w-side),side,side);break
            if crop is None:
                side=min(h,w);crop=((h-side)//2,(w-side)//2,side,side)
            pixels=VF.resized_crop(pixels,*crop,[224,224],InterpolationMode.BILINEAR,antialias=True)
            if rng.random()<.5:pixels=VF.hflip(pixels)
        else:pixels=VF.resize(pixels,[224,224],InterpolationMode.BILINEAR,antialias=True)
        pixels=VF.normalize(pixels,CACHE_SETTINGS['vision_mean'],CACHE_SETTINGS['vision_std'])
        bank=np.load(directory/'fbank.npy',mmap_mode='r');windows=self.settings['audio_windows'];length=1024
        pad=max(length*windows-len(bank),length-len(bank)%length)
        total=(len(bank)+pad)//length;parts=np.array_split(np.arange(total),windows)
        choices=[int(rng.choice(part)) if training else int(part[(len(part)-1)//2]) for part in parts]
        chunks=[]
        for i in choices:
            part=np.array(bank[i*length:(i+1)*length],copy=True)
            if len(part)<length:part=np.pad(part,((0,length-len(part)),(0,0)))
            chunks.append(torch.from_numpy(part))
        token_row=row['token_rows'][caption_index]
        return dict(ids=identifier,raw_captions=row['captions'][caption_index],vision_pixels=pixels,
            audio_spectrograms=torch.stack(chunks),caption_tokens=dict(input_ids=torch.tensor(self.tokens[token_row].copy(),dtype=torch.long),
            attention_mask=torch.tensor(self.masks[token_row].copy(),dtype=torch.long)))


def collate(rows):
    return edict(ids=[r['ids'] for r in rows],raw_captions=[r['raw_captions'] for r in rows],
        vision_pixels=torch.stack([r['vision_pixels'] for r in rows]),audio_spectrograms=torch.stack([r['audio_spectrograms'] for r in rows]),
        caption_tokens=edict({k:torch.stack([r['caption_tokens'][k] for r in rows]) for k in ('input_ids','attention_mask')}))


class GlobalBatchSampler(Sampler):
    """Identical shuffled global batches regardless of rank count; no padding IDs."""
    def __init__(self,size,rank=0,world=1,batch=64,seed=50,epoch=0,training=True):
        if batch%world:raise ValueError('Global batch must divide rank count')
        self.size,self.rank,self.world,self.batch,self.seed,self.epoch,self.training=size,rank,world,batch,seed,epoch,training
    def __iter__(self):
        if self.training:
            order=torch.randperm(self.size,generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
            width=self.batch//self.world
            for start in range(0,self.size-self.batch+1,self.batch):yield order[start+self.rank*width:start+(self.rank+1)*width]
        else:
            order=list(range(self.rank,self.size,self.world))
            for start in range(0,len(order),8):yield order[start:start+8]
    def __len__(self):return self.size//self.batch if self.training else math.ceil(len(range(self.rank,self.size,self.world))/8)


def main():
    p=argparse.ArgumentParser();roots_args(p);p.add_argument('--stage',choices=['prepare','cache'],required=True)
    p.add_argument('--dataset',choices=['all',*DATASETS],default='all');p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int,default=0)
    args=p.parse_args();roots=get_roots(args)
    if args.stage=='prepare':prepare(roots)
    else:build_cache(roots,DATASETS if args.dataset=='all' else [args.dataset],args.workers,args.limit)

if __name__=='__main__':main()
