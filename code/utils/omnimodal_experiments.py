"""Configuration, selection and memory accounting for the bounded experiment."""
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path


WORKSPACE = Path('/home/xiaobo.xia/YuxuanGao')
DATA = Path('/data0/xiaobo.xia/yuxuan.gao/PMRL')
PROJECT = WORKSPACE / 'PMRL-minimal-hyperbolic'
VIDEO_DATASETS = ('msrvtt', 'didemo', 'activitynet', 'vatex')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    resolved = path.resolve()
    if not (resolved.is_relative_to(WORKSPACE) or resolved.is_relative_to(DATA)):
        raise ValueError(f'Output outside authorized roots: {resolved}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def spec(name, phase, steps=1360, base=None, **changes):
    result = dict(temperature=.07, new_lr=1e-4, scale_multiplier=1,
                  curvature_init=1., learn_curvature=True, mapping='scaled_expmap',
                  score='spectral', consensus_weight=0., seed=50, model_type='lorentz_omnimodal')
    if base:
        result.update({k: v for k, v in base.items() if k in result})
    result.update(changes)
    result.update(name=name, phase=phase, steps=steps)
    return result


def initial_specs():
    result = []
    for index, (temperature, lr, scale) in enumerate(itertools.product((.03, .07, .15), (3e-5, 1e-4), (1, 4))):
        result.append(spec(f'screen_{index:02d}', 'screen', 340,
                           temperature=temperature, new_lr=lr, scale_multiplier=scale))
    return result


def stages_for(trial):
    if trial['phase'] == 'smoke_consensus':
        return ['train']
    stages = ['train', 'evaluation']
    if trial['steps'] == 1360 or trial['phase'] == 'smoke':
        stages.append('audio_evaluation')
    if trial['phase'] == 'smoke':
        stages.append('native_evaluation')
    return stages


def load_video_metrics(path):
    metrics = read_json(path)
    rows = {}
    for task, values in metrics.items():
        dataset = task.split('--')[-1].removesuffix('_ret')
        if dataset not in VIDEO_DATASETS:
            continue
        row = {}
        for stage in ('itc', 'itm'):
            hits = [v for k, v in values.items() if '_ret_' + stage + '_' in k]
            if len(hits) != 1:
                raise ValueError(f'Ambiguous/missing {dataset}/{stage} metrics in {path}')
            row[stage] = hits[0]
            for field in ('video_r1', 'txt_r1'):
                if not math.isfinite(row[stage][field]):
                    raise ValueError('Nonfinite retrieval metric')
        rows[dataset] = row
    if set(rows) != set(VIDEO_DATASETS):
        raise ValueError(f'Incomplete four-dataset evaluation: {path}')
    return {**{stage + '_macro_r1': sum(rows[d][stage][f] for d in VIDEO_DATASETS
                for f in ('video_r1', 'txt_r1')) / 8 for stage in ('itc', 'itm')}, 'datasets': rows}


def load_audio_metrics(path):
    rows = {}
    for task, values in read_json(path).items():
        hits = [v for k, v in values.items() if '_ret_itc_' in k]
        if len(hits) != 1 or any('_ret_itm_' in k for k in values):
            raise ValueError('Audio evaluation must contain exactly one direct-retrieval score and no ITM')
        rows[task.split('--')[-1]] = hits[0]
    if set(rows) != {'audiocaps_ret', 'clothov2_ret'}:
        raise ValueError('Incomplete audio evaluation')
    return {'macro_r1': sum(v[f] for v in rows.values() for f in ('video_r1', 'txt_r1')) / 4, 'datasets': rows}


def selection_key(result):
    return (result['video']['itm_macro_r1'], result['video']['itc_macro_r1'],
            result.get('audio', {}).get('macro_r1', -math.inf))


def competitive(result, baselines):
    video = result['video']
    for reference in baselines:
        if video['itm_macro_r1'] < reference['itm_macro_r1'] - 1:
            return False
        for dataset in VIDEO_DATASETS:
            current = video['datasets'][dataset]['itm']
            previous = reference['datasets'][dataset]['itm']
            if (current['video_r1'] + current['txt_r1']) / 2 < (previous['video_r1'] + previous['txt_r1']) / 2 - 2:
                return False
    return True


def memory_admits(free_gib, reservations, budget, headroom=8.):
    """Reservations are (budget,current_usage); unallocated peaks stay reserved."""
    outstanding = sum(max(0., total - current) for total, current in reservations)
    return free_gib - outstanding >= budget + headroom


def stage_output(suite, name, stage, attempt):
    return Path(suite) / 'trials' / name / stage / f'attempt_{attempt:02d}'


def config_for(suite, trial, stage, attempt, memory_cap):
    suite = Path(suite)
    protocol = read_json(suite / 'protocol.json')
    logs = Path(protocol['logs']) / trial['name'] / stage / f'attempt_{attempt:02d}'
    source = suite / 'source_snapshot'
    output = stage_output(suite, trial['name'], stage, attempt)
    key = 'train' if stage == 'train' else ('audio_evaluation' if stage == 'audio_evaluation' else 'evaluation')
    cfg = copy.deepcopy(read_json(suite / 'templates' / f'{key}.json'))
    run = cfg['run_cfg']
    run.update(default=str(source / 'config/pmrl/default_run_cfg.json'), output_dir=str(output),
               log_name=trial['name'] + '_' + stage, seed=trial['seed'],
               gpu_memory_limit_gib=memory_cap, resource_profile=True, checkpoint_lock=str(suite / 'checkpoint.lock'),
               resume=False, first_eval=False, log_steps=10)
    if stage == 'train':
        run.update(num_train_steps=1360, stop_after_updates=trial['steps'], new_lr=trial['new_lr'],
                   valid_freq=1, final_checkpoint_only=True, save_optimizer=trial['steps'] == 1360,
                   save_best=False, checkpoint='', mode='training', amp_init_scale=32768.)
    else:
        trained = read_json(suite / 'trials' / trial['name'] / 'train_complete.json')
        run.update(mode='testing', checkpoint=trained['checkpoint'], pretrain_dir='',
                   save_optimizer=False, stop_after_updates=0)
    model = copy.deepcopy(read_json(suite / 'templates/train.json')['model_cfg'])
    model.update(default=str(source / 'config/pmrl/default_model_cfg.json'),
                 model_type=trial['model_type'], lambda_itm=.1, hyperbolic_entailment_weight=0.,
                 hyperbolic_entailment_edges=[], hyperbolic_entailment_parent='modality_edges',
                 hyperbolic_spectral_anchor='text', hyperbolic_mapping=trial['mapping'],
                 hyperbolic_score=trial['score'], hyperbolic_temperature=trial['temperature'],
                 hyperbolic_scale_multiplier=trial['scale_multiplier'],
                 hyperbolic_curvature_init=trial['curvature_init'],
                 hyperbolic_learn_curvature=trial['learn_curvature'],
                 hyperbolic_consensus_weight=trial['consensus_weight'],
                 hyperbolic_subset_seed=trial['seed'], ret_pairwise_stage1_train=False)
    if stage != 'train':
        model.update(checkpointing=False, ret_bidirection_evaluation=True, itm_rerank_num=50,
                     evaluation_type='evaluation_mm', ret_stage1_only=stage == 'audio_evaluation',
                     ret_tv_only_candidates=stage == 'evaluation',
                     ret_native_tuple_candidates=stage == 'native_evaluation',
                     ret_skip_euclidean_diagnostics=trial['model_type'] == 'lorentz_omnimodal')
    cfg['model_cfg'] = model
    if stage == 'train':
        cfg['data_cfg']['train'][0].update(drop_subtitles=True, n_workers=4)
    for split in ('train', 'val'):
        for data in cfg['data_cfg'][split]:
            data['sample_trace_dir'] = str(logs / 'media_trace')
            if stage != 'train' and trial['phase'] == 'smoke':
                dataset = data['name'].removesuffix('_ret')
                data['txt'] = str(suite / 'smoke_manifests' / f'{dataset}.json')
            for field in ('txt', 'vision', 'audio', 'video_overrides'):
                if field in data:
                    actual = Path(data[field]).resolve()
                    if not actual.exists() or not (actual.is_relative_to(DATA) or actual.is_relative_to(WORKSPACE)):
                        raise ValueError(f'Unavailable or unauthorized input: {actual}')
    output.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    (logs / 'internal').mkdir()
    (output / 'log').symlink_to(logs / 'internal')
    cfg_path = suite / 'trials' / trial['name'] / 'configs' / f'{stage}_{attempt:02d}.json'
    write_json(cfg_path, cfg)
    return cfg_path, output, logs
