"""Portable paths and immutable protocol for the paired low-dimensional suite."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path('/data-nfs/xiaoboxia/YuxuanGao')
DEFAULT_SUITE = DEFAULT_DATA / 'lowdim_finetune_20261019'
METHODS = ('vast', 'gram', 'pmrl', 'hyp_cone', 'hyp_none', 'euc_cone', 'euc_none')
DIMENSIONS = (8, 32, 128, 512)
DATASETS = ('msrvtt', 'activitynet')
DATA_SETTINGS = {
    'msrvtt': dict(vision_frames=2, audio_windows=1, max_caption_len=40),
    'activitynet': dict(vision_frames=8, audio_windows=2, max_caption_len=160),
}
CACHE_SETTINGS = dict(version=1, candidate_frames=32, short_edge=256, resize='bilinear',
    audio_melbins=64, audio_target_length=1024, audio_mean=15.41663, audio_std=6.55582,
    sample_rate=16000, frame_length=25, frame_shift=10,
    vision_mean=[.48145466,.4578275,.40821073], vision_std=[.26862954,.26130258,.27577711])


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):h.update(block)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def append_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')


def roots_args(parser):
    parser.add_argument('--workspace',type=Path,default=ROOT)
    parser.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--suite',type=Path,default=DEFAULT_SUITE)
    parser.add_argument('--log-root',type=Path,default=None)
    parser.add_argument('--author-root',type=Path,default=None)


def get_roots(args):
    workspace=args.workspace.resolve(); data=args.data_root.resolve(); suite=args.suite.resolve()
    log=(args.log_root or workspace/'logs'/suite.name).resolve()
    if not log.is_relative_to(workspace):raise ValueError('Logs must remain under the selected workspace')
    if not (suite.is_relative_to(data) or suite.is_relative_to(workspace)):
        raise ValueError('Suite must be under the explicit workspace/data root')
    authors=(args.author_root or workspace.parent/'main_result_baselines').resolve()
    return dict(workspace=workspace,data=data,suite=suite,logs=log,authors=authors)


def path_ref(path, roots):
    path=Path(path).resolve()
    for key in ('data','workspace','suite','authors'):
        if path.is_relative_to(roots[key]):return dict(root=key,path=str(path.relative_to(roots[key])))
    raise ValueError(f'Resource is outside explicit roots: {path}')


def resolve_ref(ref, roots):
    if ref['root'] not in ('data','workspace','suite','authors'):raise ValueError('Unknown path root')
    p=(roots[ref['root']]/ref['path']).resolve()
    if not any(p.is_relative_to(roots[k]) for k in ('data','workspace','suite','authors')):
        raise ValueError('Mapped resource escapes explicit roots')
    return p


def trial_name(dataset,method,dimension,seed=50):return f'{dataset}_{method}_d{dimension}_s{seed}'


def geometry_for(method):return 'hyperbolic' if method.startswith('hyp_') else 'euclidean'


def ours(method):return method.startswith(('hyp_','euc_'))


def source_files(roots):
    """Code/assets needed by constructors, excluding checkpoints and environments."""
    result={}
    skip={'.pt','.pth','.bin','.safetensors','.ckpt','.npy','.npz','.pyc','.so'}
    for directory in ('model','utils','data'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in skip:
                result['code/'+str(path.relative_to(ROOT))]=path
    for name in ['lowdim_common.py','lowdim_data.py','lowdim_models.py','lowdim_finetune.py','lowdim_schedule.py','lowdim_verify.py','lowdim_report.py','lowdim_handoff.py']:
        path=ROOT/'experiments/subset_hierarchy'/name;result['code/'+str(path.relative_to(ROOT))]=path
    for path in (ROOT/'tests').glob('*lowdim*.py'):
        result['code/'+str(path.relative_to(ROOT))]=path
    p=read_json(roots['suite']/'protocol.json')
    for method in ('vast','gram'):
        source=resolve_ref(p['author_sources'][method],roots)
        for path in (source/'model').rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in skip:
                result['authors/'+method.upper()+'/'+str(path.relative_to(source))]=path
        for filename in ('LICENSE','NOTICE'):
            if (source/filename).is_file():result['authors/'+method.upper()+'/'+filename]=source/filename
        if method=='gram':result['authors/GRAM/utils/volume.py']=source/'utils/volume.py'
    return result


def source_manifest(roots):return {key:file_hash(path) for key,path in sorted(source_files(roots).items())}


def runtime_versions():
    return {name:importlib.metadata.version(name) for name in ('torch','torchvision','torchaudio','numpy','transformers')}


def require_cpu_gate(roots):
    gate=read_json(roots['suite']/'CPU_READY.json')
    if not gate.get('passed') or gate['protocol_sha256']!=file_hash(roots['suite']/'protocol.json'):
        raise RuntimeError('CPU readiness gate/protocol mismatch')
    if gate.get('runtime_versions')!=runtime_versions():raise RuntimeError('Runtime packages changed; repeat CPU and GPU validation')
    if gate['training_sources']!=source_manifest(roots):raise RuntimeError('Training code changed after CPU validation')
    for name in DATASETS:
        if gate['data_ready_sha256'][name]!=file_hash(roots['suite']/'data'/name/'READY.json'):
            raise RuntimeError('Data readiness changed after CPU validation')
    return gate


def require_gpu_gate(roots):
    cpu=require_cpu_gate(roots);gate=read_json(roots['suite']/'GPU_READY.json')
    if not gate.get('passed') or gate['cpu_ready_sha256']!=file_hash(roots['suite']/'CPU_READY.json'):
        raise RuntimeError('GPU gate does not match the validated CPU suite')
    for relative,expected in gate['evidence_sha256'].items():
        path=(roots['suite']/relative).resolve()
        if not path.is_relative_to(roots['suite']) or file_hash(path)!=expected:
            raise RuntimeError('GPU gate evidence changed')
    return gate


def protocol_base():
    return dict(schema_version=1,datasets=DATA_SETTINGS,methods=list(METHODS),dimensions=list(DIMENSIONS),
        seed=50,epochs=10,global_batch=64,learning_rate=2e-5,vision_learning_rate=5e-7,
        betas=[.9,.98],weight_decay=.01,warmup_fraction=.05,grad_clip=2.,
        cache=CACHE_SETTINGS,temperature=.07,curvature=1.,hyperbolic_K=.1,euclidean_K=.2,
        radius_min=.21,radius_max=4.,entailment_weight=.1,pair_mode='minimal9',
        early_stopping=False,selection='Validation six-direction macro R@1; epoch zero eligible; ties keep earliest.',
        initialization='Pretrained encoders; train-only nested PCA; identical VAST-initialized seven heads and radius=1 for all four geometry arms.',
        inference='Six directions, no ITM: native cosine / hyperbolic geodesic / Euclidean distance.',
        execution_policy='CPU preparation only until explicit GPU-stage launch; never auto-launch on newly available GPUs.')
