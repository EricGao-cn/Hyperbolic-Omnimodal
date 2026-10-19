"""Opt-in profiling and resource-bounded scheduler. Default is read-only status."""
import argparse
import fcntl
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.subset_hierarchy.lowdim_common import *

FAMILIES=('vast','gram','pmrl','hyp_cone')

def family(method):return 'hyp_cone' if ours(method) else method


def gpu_inventory(devices):
    result=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    rows=[]
    for line in result.stdout.splitlines():
        index,name,total,free,util=[x.strip() for x in line.split(',')]
        if index in devices:rows.append(dict(index=index,name=name,total_gib=float(total)/1024,free_gib=float(free)/1024,utilization=float(util)))
    if {r['index'] for r in rows}!=set(devices):raise ValueError('Requested GPU index is unavailable')
    return rows


def hardware_key(inventory):
    # Profiles are transferable between identical cards, but not GPU models.
    return sorted({(r['name'],round(r['total_gib'],1)) for r in inventory})


def allocate(inventory,world,peak,active,dual=False):
    """Reserve measured peak plus 3 GiB per GPU; at most two own ranks/card."""
    choices=[]
    for row in inventory:
        jobs=[a for a in active if row['index'] in a['devices']]
        if len(jobs)>=(2 if dual else 1):continue
        # Pending processes may not yet have allocated their memory. Subtract
        # all reservations conservatively even when nvidia-smi already sees it.
        available=row['free_gib']-sum(a['peak_gib'] for a in jobs)
        if available>=peak+3:choices.append((len(jobs),-available,row['index']))
    if len(choices)<world:return None
    return [x[2] for x in sorted(choices)[:world]]


def root_cli(roots):
    return sum(([ '--'+{'workspace':'workspace','data':'data-root','suite':'suite','logs':'log-root','authors':'author-root'}[key],str(value)] for key,value in roots.items()),[])


def launch(roots,stage,dataset,method,dimension,devices,tag,steps=12,cap=0.,resume=False):
    name=trial_name(dataset,method,dimension);log=roots['logs']/'scheduler';log.mkdir(parents=True,exist_ok=True)
    path=log/(name+'_'+stage+'_'+tag+'.log');stream=path.open('a')
    command=[sys.executable,'-u','-B',str(ROOT/'experiments/subset_hierarchy/lowdim_finetune.py'),*root_cli(roots),
        '--stage',stage,'--dataset',dataset,'--method',method,'--dimension',str(dimension),'--devices',','.join(devices),
        '--world-size',str(len(devices)),'--allow-gpu','--steps',str(steps),'--tag',tag,'--memory-cap-gib',str(cap)]
    if resume:command.append('--resume')
    env=dict(os.environ,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=','.join(devices))
    proc=subprocess.Popen(command,cwd=roots['workspace'],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);stream.close()
    append_json(log/'processes.jsonl',dict(pid=proc.pid,command=command,devices=devices,log=str(path),time=time.time()))
    return proc,path


def run_wait(*args,**kwargs):
    proc,log=launch(*args,**kwargs)
    try:code=proc.wait()
    except BaseException:
        if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM);proc.wait()
        raise
    return code,log


def report_path(roots,stage,d,m,w,world,tag):
    return roots['suite']/'profiles'/trial_name(d,m,w)/(f'{stage}_w{world}_'+tag)/'DONE.json'


def profile_all(roots,devices):
    records=[];hardware=hardware_key(gpu_inventory(devices));stamp=str(time.time_ns())
    for dataset,method in itertools.product(DATASETS,FAMILIES):
        for world in (1,2,4,8):
            if world>len(devices):continue
            inv=gpu_inventory(devices);group=allocate(inv,world,6,[])
            if group is None:continue
            tag='solo_'+stamp
            code,log=run_wait(roots,'profile',dataset,method,512,group,tag)
            if code:
                append_json(roots['logs']/'scheduler/profile_failures.jsonl',dict(dataset=dataset,method=method,world=world,log=str(log),exit_code=code));continue
            path=report_path(roots,'profile',dataset,method,512,world,tag);report=read_json(path)
            peak=max(r['peak_reserved_gib'] for r in report['ranks'])
            record=dict(dataset=dataset,family=method,world_size=world,profile_devices=group,initial_inventory=inv,peak_gib=peak,samples_per_second=report['samples_per_second'],
                solo_evidence=str(path.relative_to(roots['suite'])),dual=False)
            # Test two jobs on the same group only when both measured peaks fit.
            inv={r['index']:r for r in gpu_inventory(devices)}
            if all(inv[g]['free_gib']>=2*peak+3 for g in group):
                cap=min((inv[g]['free_gib']-3)/2 for g in group)
                processes=[];pair_paths=[]
                try:
                    for i in range(2):
                        pair_tag=f'dual{i}_{stamp}';processes.append(launch(roots,'profile',dataset,method,512,group,pair_tag,cap=cap)[0])
                        pair_paths.append(report_path(roots,'profile',dataset,method,512,world,pair_tag))
                    codes=[p.wait() for p in processes]
                finally:
                    for p in processes:
                        if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait()
                if all(c==0 for c in codes):
                    paired=[read_json(p) for p in pair_paths];throughput=sum(p['samples_per_second'] for p in paired)
                    if throughput>=1.1*report['samples_per_second']:
                        record.update(dual=True,dual_samples_per_second=throughput,
                            peak_gib=max(peak,*[r['peak_reserved_gib'] for p in paired for r in p['ranks']]),
                            dual_evidence=[str(p.relative_to(roots['suite'])) for p in pair_paths])
            records.append(record)
            write_json(roots['suite']/'layouts.json',dict(hardware=hardware,records=records,cpu_ready_sha256=file_hash(roots['suite']/'CPU_READY.json')))
    missing=[(d,m) for d,m in itertools.product(DATASETS,FAMILIES) if not any(r['dataset']==d and r['family']==m for r in records)]
    if missing:raise RuntimeError(f'No feasible profile for {missing}; see profile failure logs')


def load_layouts(roots,devices):
    saved=read_json(roots['suite']/'layouts.json')
    if saved['hardware']!=[list(x) for x in hardware_key(gpu_inventory(devices))]:raise RuntimeError('GPU hardware changed; repeat profiling')
    if saved['cpu_ready_sha256']!=file_hash(roots['suite']/'CPU_READY.json'):raise RuntimeError('CPU suite changed; repeat profiling')
    return saved['records']


def options(records,dataset,method,world=None):
    rows=[r for r in records if r['dataset']==dataset and r['family']==family(method) and (world is None or r['world_size']==world)]
    return sorted(rows,key=lambda r:r.get('dual_samples_per_second',r['samples_per_second'])/r['world_size'],reverse=True)


def preflight(roots,devices):
    records=load_layouts(roots,devices);stamp=str(time.time_ns())
    for d,m,w in itertools.product(DATASETS,METHODS,(8,512)):
        choice=options(records,d,m)[0];group=allocate(gpu_inventory(devices),choice['world_size'],choice['peak_gib'],[])
        if group is None:raise RuntimeError('Insufficient current free GPU memory; rerun preflight when available')
        code,log=run_wait(roots,'smoke',d,m,w,group,'gate_'+stamp)
        if code:raise RuntimeError(f'Smoke failed: {log}')
    seal_gpu_gate(roots)


def seal_gpu_gate(roots):
    require_cpu_gate(roots);evidence={};heads={}
    for d,m in itertools.product(DATASETS,('vast','gram','pmrl')):
        for filename in (m+'.pt',m+'_READY.json'):
            path=roots['suite']/'bases'/d/filename;evidence[str(path.relative_to(roots['suite']))]=file_hash(path)
        ready=read_json(roots['suite']/'bases'/d/(m+'_READY.json'))
        if not ready['passed'] or ready['sha256']!=evidence[f'bases/{d}/{m}.pt']:raise RuntimeError('Invalid PCA gate')
    source=digest(source_manifest(roots))
    for d,m,w in itertools.product(DATASETS,METHODS,(8,512)):
        paths=sorted((roots['suite']/'profiles'/trial_name(d,m,w)).glob('smoke_w*/DONE.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        valid=[]
        for path in paths:
            row=read_json(path);cfg=row['config']
            if row.get('passed') and row['steps']>=2 and row['global_batch']==64 and cfg['source']==source and cfg['protocol_sha256']==file_hash(roots['suite']/'protocol.json') and cfg['data']==read_json(roots['suite']/'data'/d/'READY.json'):valid.append((path,row))
        if not valid:raise RuntimeError(f'Missing matching full-encoder smoke: {d}/{m}/{w}')
        path,row=valid[0];evidence[str(path.relative_to(roots['suite']))]=file_hash(path)
        if ours(m):heads.setdefault((d,w),set()).add(row['config']['initialization']['heads_sha256'])
    if any(len(v)!=1 for v in heads.values()):raise RuntimeError('Four geometry arms do not share initial heads')
    layout=roots['suite']/'layouts.json';evidence['layouts.json']=file_hash(layout)
    for row in read_json(layout)['records']:
        for relative in [row['solo_evidence'],*row.get('dual_evidence',[])]:evidence[relative]=file_hash(roots['suite']/relative)
    write_json(roots['suite']/'GPU_READY.json',dict(passed=True,cpu_ready_sha256=file_hash(roots['suite']/'CPU_READY.json'),evidence_sha256=evidence,
        checks=['six train-only PCA bases','28 full-encoder D8/D512 smokes','finite gradients and nonzero encoder/head branches','shared initial heads','measured resource layouts']))


def train_all(roots,devices):
    require_gpu_gate(roots);records=load_layouts(roots,devices);active=[]
    queue=sorted(read_json(roots['suite']/'tasks.json'),key=lambda t:(t['dataset']!='activitynet',-t['dimension']))
    failed=[]
    try:
        while queue or active:
            for item in list(active):
                code=item['proc'].poll()
                if code is not None:
                    active.remove(item)
                    if code:failed.append(dict(task=item['task'],log=str(item['log']),exit_code=code))
            inventory=gpu_inventory(devices)
            for task in list(queue):
                run=roots['suite']/'runs'/task['name']
                if (run/'DONE.json').exists():
                    from experiments.subset_hierarchy.lowdim_finetune import validate_completed
                    validate_completed(roots,run);queue.remove(task);continue
                existing=read_json(run/'config.json')['world_size'] if (run/'config.json').exists() else None
                choices=options(records,task['dataset'],task['method'],existing)[:1]
                if not choices:raise RuntimeError(f'No compatible profile for resume: {task["name"]}')
                for choice in choices:
                    world=choice['world_size']
                    if sum(len(a['devices']) for a in active)+world>16:continue
                    # Shared-card execution requires every resident job to have
                    # passed the same paired-family profile.
                    allowed=[a for a in active if not (choice['dual'] and a['layout_key']==(task['dataset'],family(task['method']),world))]
                    blocked={g for a in allowed for g in a['devices']}
                    candidates=[r for r in inventory if r['index'] not in blocked]
                    group=allocate(candidates,world,choice['peak_gib']+.5,active,choice['dual'])
                    if group is None:continue
                    cap=choice['peak_gib']+.5
                    proc,log=launch(roots,'train',task['dataset'],task['method'],task['dimension'],group,'formal',cap=cap,resume=True)
                    active.append(dict(proc=proc,log=log,task=task,devices=group,peak_gib=cap,layout_key=(task['dataset'],family(task['method']),world)))
                    queue.remove(task);break
            write_json(roots['logs']/'scheduler/status.json',dict(queued=[t['name'] for t in queue],running=[dict(task=a['task']['name'],pid=a['proc'].pid,devices=a['devices']) for a in active],failed=failed))
            if queue or active:time.sleep(10)
    finally:
        for item in active:
            if item['proc'].poll() is None:os.killpg(item['proc'].pid,signal.SIGTERM);item['proc'].wait()
    if failed:raise RuntimeError(f'{len(failed)} runs failed; inspect scheduler/status.json')


def main():
    p=argparse.ArgumentParser();roots_args(p);p.add_argument('--stage',choices=['status','basis','profile','preflight','train','all'],default='status')
    p.add_argument('--execute-gpu',action='store_true');p.add_argument('--devices',default='');args=p.parse_args();roots=get_roots(args)
    if args.stage=='status':
        print(json.dumps(dict(cpu_ready=(roots['suite']/'CPU_READY.json').exists(),gpu_ready=(roots['suite']/'GPU_READY.json').exists(),
            completed=len(list((roots['suite']/'runs').glob('*/DONE.json'))),total=56,automatic_gpu_launch=False),indent=2));return
    if not args.execute_gpu:p.error('Execution requires explicit --execute-gpu')
    devices=[d.strip() for d in args.devices.split(',') if d.strip()]
    if not devices or len(set(devices))!=len(devices) or any(not d.isdigit() for d in devices):p.error('Supply distinct numeric GPU indices')
    require_cpu_gate(roots);gpu_inventory(devices)
    with (roots['suite']/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stages=['basis','profile','preflight','train'] if args.stage=='all' else [args.stage]
        for stage in stages:
            if stage=='basis':
                for d,m in itertools.product(DATASETS,('vast','gram','pmrl')):
                    marker=roots['suite']/'bases'/d/(m+'_READY.json')
                    if marker.exists() and read_json(marker).get('passed'):continue
                    inv=gpu_inventory(devices);group=None
                    for world in (8,4,2,1):
                        group=allocate(inv,world,6,[])
                        if group is not None:break
                    if group is None:raise RuntimeError('No GPU with 9 GiB free for frozen extraction')
                    code,log=run_wait(roots,'basis',d,m,512,group,'basis')
                    if code:raise RuntimeError(f'PCA extraction failed: {log}')
            elif stage=='profile':profile_all(roots,devices)
            elif stage=='preflight':preflight(roots,devices)
            else:train_all(roots,devices)

if __name__=='__main__':main()
