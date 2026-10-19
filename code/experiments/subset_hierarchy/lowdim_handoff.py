"""Small GitHub-ready source/annotation package; rebuild caches on the target host.

Export is local only. Bootstrap uses only the exported annotation cohorts and
explicit target-host resource paths, never source-host caches or old experiments.
"""
import argparse
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.subset_hierarchy.lowdim_common import *


def export_code(roots,destination):
    destination=destination.resolve()
    if not destination.is_relative_to(roots['workspace']):raise ValueError('Local handoff must remain under workspace')
    if destination.exists():raise FileExistsError('Choose a new handoff folder; existing files are preserved')
    destination.mkdir(parents=True)
    for relative,source in source_files(roots).items():
        path=destination/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,path)
    for name in ('requirements-lowdim-finetune.txt','LICENSE'):
        if (ROOT/name).exists():shutil.copy2(ROOT/name,destination/'code'/name)
    doc=ROOT/'experiments/subset_hierarchy/LOWDIM_README.md'
    shutil.copy2(doc,destination/'code/experiments/subset_hierarchy/LOWDIM_README.md')
    shutil.copy2(ROOT/'experiments/subset_hierarchy/LOWDIM_SERVER_HANDOFF.md',destination/'README.md')
    shutil.copy2(ROOT/'experiments/subset_hierarchy/LOWDIM_SERVER_HANDOFF.md',destination/'code/experiments/subset_hierarchy/LOWDIM_SERVER_HANDOFF.md')
    for name in ('protocol.json','tasks.json'):shutil.copy2(roots['suite']/name,destination/name)
    summaries={}
    for dataset in DATASETS:
        source=roots['suite']/'data'/dataset/'candidates.json'
        dest=destination/'annotations'/dataset/'candidates.json';dest.parent.mkdir(parents=True)
        shutil.copy2(source,dest);rows=read_json(dest)
        summaries[dataset]=dict(counts={s:len(r) for s,r in rows.items()},sha256=file_hash(dest))
    write_json(destination/'annotations/manifest.json',summaries)
    (destination/'.gitignore').write_text('.venv/\n**/__pycache__/\n.cache/\ncode/.cache/\ncode/.tmp/\ncode/logs/\ncode/outputs/\ncode/handoff/\n*.pt\n*.pth\n*.bin\n*.npy\n*.npz\n*.wav\n*.mp4\n')
    # BPE tokenizer .gz files are small required source assets and are included.
    write_json(destination/'SHA256.json',{str(p.relative_to(destination)):file_hash(p) for p in sorted(destination.rglob('*')) if p.is_file()})
    files=[p for p in destination.rglob('*') if p.is_file()]
    print('GitHub-ready local folder:',destination)
    print('Files:',len(files),'total bytes:',sum(p.stat().st_size for p in files),'largest bytes:',max(p.stat().st_size for p in files))
    print('No git commit, push, upload, media cache, model checkpoint or readiness gate included.')


def bootstrap(args,roots):
    package=args.package.resolve();suite=roots['suite']
    manifest=read_json(package/'SHA256.json')
    # Validate only package metadata used for this operation. Full source checks
    # are performed by CPU validation after the destination environment is ready.
    for relative in ['protocol.json','tasks.json','annotations/manifest.json',*[f'annotations/{d}/candidates.json' for d in DATASETS]]:
        if file_hash(package/relative)!=manifest[relative]:raise ValueError('Handoff annotation/protocol checksum mismatch: '+relative)
    p=read_json(package/'protocol.json');p['model_cfg'].pop('default',None)
    p['gram_hps']={'model_cfg':p['gram_hps']['model_cfg']};p['gram_hps']['model_cfg'].pop('default',None)
    for method in ('vast','gram','pmrl'):
        path=getattr(args,method+'_checkpoint').resolve()
        if not path.is_file():raise FileNotFoundError(path)
        p['checkpoints'][method]=path_ref(path,roots)
    if not args.assets.is_dir():raise FileNotFoundError(args.assets)
    p['assets']=path_ref(args.assets,roots)
    p['author_sources']={m:path_ref(roots['authors']/m.upper(),roots) for m in ('vast','gram')}
    for method in ('vast','gram'):
        if not (roots['authors']/method.upper()/'model'/f'{method}.py').is_file():raise FileNotFoundError('Missing bundled author source: '+method)
    planned={};missing=[]
    for dataset in DATASETS:
        video=getattr(args,dataset+'_video').resolve();audio=getattr(args,dataset+'_audio').resolve()
        rows=read_json(package/'annotations'/dataset/'candidates.json')
        identifiers=[r['clip_id'] for values in rows.values() for r in values]
        if len(identifiers)!=len(set(identifiers)):raise ValueError('Duplicate or overlapping annotation IDs')
        from re import fullmatch
        for identifier in identifiers:
            if not fullmatch(r'[A-Za-z0-9_.@-]+',identifier) or identifier in ('.','..'):raise ValueError('Unsafe annotation ID')
            if not (video/(identifier+'.mp4')).is_file():missing.append(f'{dataset}: video {identifier}.mp4')
            if not (audio/(identifier+'.wav')).is_file():missing.append(f'{dataset}: audio {identifier}.wav')
        planned[dataset]=dict(candidates=rows,sources=dict(video=path_ref(video,roots),audio=path_ref(audio,roots)))
    if missing:raise RuntimeError(f'{len(missing)} required media files missing; do not silently change the cohort. First 20: {missing[:20]}')
    if (suite/'protocol.json').exists() and read_json(suite/'protocol.json')!=p:raise ValueError('Destination suite already has a different protocol; choose a new suite')
    # An identical bootstrap is idempotent, and never alters completed caches.
    for dataset,value in planned.items():
        for name,content in value.items():
            target=suite/'data'/dataset/(name+'.json')
            if target.exists() and read_json(target)!=content:raise ValueError('Existing destination annotation/source mapping differs')
    suite.mkdir(parents=True,exist_ok=True)
    write_json(suite/'protocol.json',p);write_json(suite/'tasks.json',read_json(package/'tasks.json'))
    for dataset,value in planned.items():
        for name,content in value.items():write_json(suite/'data'/dataset/(name+'.json'),content)
    write_json(suite/'BOOTSTRAP.json',dict(passed=True,annotation_manifest=read_json(package/'annotations/manifest.json'),
        cpu_ready=False,gpu_ready=False,checkpoint_hash_verification='pending CPU model audit'))
    print('Target suite initialized:',suite)
    print('Next: local cache generation, CPU validation, then explicitly authorized GPU stages. No GPU was launched.')


def main():
    p=argparse.ArgumentParser();roots_args(p);p.add_argument('stage',choices=['export','bootstrap']);p.add_argument('--destination',type=Path)
    p.add_argument('--package',type=Path)
    for method in ('vast','gram','pmrl'):p.add_argument('--'+method+'-checkpoint',type=Path)
    p.add_argument('--assets',type=Path)
    for dataset in DATASETS:
        for kind in ('video','audio'):p.add_argument('--'+dataset+'-'+kind,type=Path)
    args=p.parse_args();roots=get_roots(args)
    if args.stage=='export':export_code(roots,args.destination or roots['workspace']/'handoff'/('lowdim_github_'+str(time.time_ns())))
    else:
        required=['package','vast_checkpoint','gram_checkpoint','pmrl_checkpoint','assets',*[d+'_'+k for d in DATASETS for k in ('video','audio')]]
        for key in required:
            if getattr(args,key) is None:p.error('Missing --'+key.replace('_','-'))
        bootstrap(args,roots)

if __name__=='__main__':main()
