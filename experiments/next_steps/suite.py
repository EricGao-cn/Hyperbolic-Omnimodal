"""Gated experiment suite for the 2026-10-13 plan (see PLAN.md in this folder).

One suite directory holds the protocol, a frozen source snapshot, every trial
spec, generated configs, outputs and DONE/FAILED markers. Selection always uses
the held-out VAST split; test numbers are only reported for the chosen trial.

    python experiments/next_steps/suite.py init --suite S --templates T --pretrain-dir P \
        --holdout-dir H                      # H from holdout_split.py
    python experiments/next_steps/suite.py add gate_a
    python experiments/next_steps/suite.py run --gpus 0,1,2,3
    python experiments/next_steps/suite.py add gate_a_promote      # top-2 screens -> full
    python experiments/next_steps/suite.py add gate_a_contrast     # symmetric vs text_vs_va
    python experiments/next_steps/suite.py run --gpus 0,1,2,3
    python experiments/next_steps/suite.py gate-a                  # writes gate_a.json
    python experiments/next_steps/suite.py add gate_b              # parents on the Gate-A winner
    ...

The suite directory is remembered in ./.next_steps_suite after init, so --suite
can be omitted afterwards. T is a directory with train.json, evaluation.json and
audio_evaluation.json (copy templates/ from the 20260929 Lorentz suite so the
media, batch and evaluation protocol match the existing results).
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import signal
import socket
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from utils.omnimodal_experiments import load_audio_metrics, load_video_metrics  # noqa: E402

POINTER = REPO / '.next_steps_suite'
FULL, SCREEN = 1360, 340
TEMPERATURES = (.03, .07, .15)
TRANSFORMS = ('distance', 'inverse', 'log')
EXTRA_SEEDS = (42, 123)

# Gate thresholds (percentage points of R@1); see PLAN.md section 4.
GATE_A = dict(top50=-.3, stage1=-1., audio=-1.)
GATE_B = dict(cone=.3, top50=-.3)

KERNEL = dict(model_type='hyperbolic_spectral_pmrl', hyperbolic_spectral_temperature=.03,
              hyperbolic_geometry_fp64=True,
              hyperbolic_spectral_anchor='text', hyperbolic_entailment_weight=0.,
              hyperbolic_entailment_edges=[], hyperbolic_entailment_parent='modality_edges',
              hyperbolic_parent_contraction=1., hyperbolic_parent_floor_margin=.001,
              hyperbolic_entailment_min_radius=.1, lambda_itm=.1, ret_pairwise_stage1_train=False)
LORENTZ = dict(model_type='lorentz_omnimodal', hyperbolic_mapping='direction_radius',
               hyperbolic_score='spectral', hyperbolic_score_transform='distance',
               hyperbolic_contrast='text_vs_va', hyperbolic_temperature=.03,
               hyperbolic_curvature_init=1., hyperbolic_learn_curvature=True,
               hyperbolic_scale_multiplier=1, hyperbolic_consensus_weight=0.,
               hyperbolic_entailment_weight=0., hyperbolic_entailment_edges=[],
               hyperbolic_entailment_parent='modality_edges', hyperbolic_parent_contraction=1.,
               hyperbolic_parent_floor_margin=.001, hyperbolic_entailment_min_radius=.1,
               lambda_itm=.1, ret_pairwise_stage1_train=False)
PMRL_GLOBAL = dict(model_type='pmrl', pmrl_global_negatives=True, tau2=.1, lambda_itm=.1,
                   ret_pairwise_stage1_train=False, log_spectral_diagnostics=True)


# ---------------------------------------------------------------- utilities

def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def suite_dir(args):
    if args.suite:
        return Path(args.suite).resolve()
    if POINTER.exists():
        return Path(POINTER.read_text().strip())
    sys.exit('No suite: pass --suite or run init first')


def load_trials(suite):
    path = suite / 'trials.json'
    return read_json(path) if path.exists() else []


def save_trials(suite, trials):
    names = [t['name'] for t in trials]
    if len(names) != len(set(names)):
        raise ValueError('duplicate trial names')
    write_json(suite / 'trials.json', trials)


def trial(name, phase, model, steps=FULL, seed=50, new_lr=1e-4, run=None, **extra):
    model = dict(model)
    if model['model_type'] == 'lorentz_omnimodal':
        model['hyperbolic_subset_seed'] = seed
    return dict(name=name, phase=phase, steps=steps, seed=seed, new_lr=new_lr,
                model=model, run=dict(run or {}), **extra)


def is_full(t):
    return t['steps'] == FULL


def stages_for(t):
    if 'stages' in t:
        return t['stages']
    stages = ['train', 'holdout_video', 'holdout_audio']
    if is_full(t):
        stages += ['video', 'audio']
        if t['model']['model_type'] == 'lorentz_omnimodal' and t['model']['hyperbolic_score'] != 'allpair':
            stages.append('native')
    return stages


def stage_dir(suite, name, stage):
    return suite / 'trials' / name / stage


def done(suite, name, stage):
    return (stage_dir(suite, name, stage) / 'DONE.json').exists()


# ---------------------------------------------------------------- init

def cmd_init(args):
    suite = Path(args.suite).resolve()
    if (suite / 'protocol.json').exists():
        sys.exit(f'{suite} already initialized')
    templates = Path(args.templates).resolve()
    for name in ('train.json', 'evaluation.json', 'audio_evaluation.json'):
        if not (templates / name).exists():
            sys.exit(f'missing template {templates / name}')
    snapshot = suite / 'source_snapshot'
    files = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'],
                           cwd=REPO, check=True, capture_output=True, text=True).stdout.splitlines()
    for rel in files:
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, target)
    weights = Path(args.weights_dir or REPO / 'pretrained_weights').resolve()
    if not weights.exists():
        sys.exit(f'pretrained weights not found: {weights}')
    (snapshot / 'pretrained_weights').symlink_to(weights)
    shutil.copytree(templates, suite / 'templates')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(['git', 'diff', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout
    protocol = dict(created=time.strftime('%Y-%m-%dT%H:%M:%S'), git_commit=commit,
                    git_diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),
                    templates_from=str(templates), pretrain_dir=str(Path(args.pretrain_dir).resolve()),
                    holdout=None, gates=dict(a=GATE_A, b=GATE_B))
    protocol['resources'] = dict(train_memory_gib=args.train_memory_gib,
                                 eval_memory_gib=args.eval_memory_gib,
                                 logs_dir=str(Path(args.logs_dir or REPO / '.tmp' / suite.name).resolve()),
                                 gpu_headroom_gib=8, disk_reserve_gib=40,
                                 save_optimizer=False)
    (suite / 'source_changes.diff').write_text(diff)
    if args.holdout_dir:
        holdout = read_json(Path(args.holdout_dir) / 'holdout_manifest.json')
        protocol['holdout'] = holdout
    else:
        print('WARNING: no --holdout-dir; selection would fall back to test metrics, which the plan forbids.')
    write_json(suite / 'protocol.json', protocol)
    POINTER.write_text(str(suite))
    print(f'initialized {suite} from commit {commit[:10]}')


# ---------------------------------------------------------------- trial sets

def add_gate_a(trials, suite, args):
    out = [trial('a_kernel_e0', 'gate_a', KERNEL),
           trial('a_m_distance_t0.03', 'gate_a', LORENTZ)]
    for transform, tau in itertools.product(TRANSFORMS, TEMPERATURES):
        out.append(trial(f'a_screen_{transform}_t{tau}', 'gate_a_screen',
                         dict(LORENTZ, hyperbolic_score_transform=transform, hyperbolic_temperature=tau),
                         steps=SCREEN))
    return out


def add_gate_a_promote(trials, suite, args):
    screens = [t for t in trials if t['phase'] == 'gate_a_screen']
    ranked = rank_by_holdout(suite, screens)
    if len(ranked) < len(screens):
        sys.exit(f'only {len(ranked)}/{len(screens)} screens have holdout metrics; refusing partial selection')
    out = []
    for t, key in ranked[:args.top]:
        m = t['model']
        name = f"a_m_{m['hyperbolic_score_transform']}_t{m['hyperbolic_temperature']}"
        out.append(trial(name, 'gate_a', m, screen_of=t['name'], selection_key=key))
    return out


def add_gate_a_contrast(trials, suite, args):
    best = best_full(suite, trials, lambda t: t['phase'] == 'gate_a'
                     and t['model']['model_type'] == 'lorentz_omnimodal'
                     and t['model']['hyperbolic_contrast'] == 'text_vs_va')
    m = dict(best['model'], hyperbolic_contrast='symmetric')
    return [trial(best['name'] + '_symmetric', 'gate_a', m, contrast_of=best['name'])]


def add_gate_b(trials, suite, args):
    winner = gate_winner(suite)
    base = dict(winner['model'])
    if base['model_type'] != 'lorentz_omnimodal':
        print('Gate A failed: parents are added to the kernel score (fallback design).')
    # The Perron parent needs the M model; after a Gate-A fallback use the tangent mean.
    parent = 'perron' if base['model_type'] == 'lorentz_omnimodal' else 'tangent_mean'
    out = [trial(f'b_{parent}_w{weight}', 'gate_b',
                 dict(base, hyperbolic_entailment_parent=parent, hyperbolic_entailment_weight=weight),
                 base_trial=winner['name'])
           for weight in (.03, .1, .3)]
    out.append(trial('b_text_w0.1', 'gate_b', dict(base, hyperbolic_entailment_weight=.1,
                     hyperbolic_entailment_edges=[['t', 'v'], ['t', 'a']]), base_trial=winner['name']))
    out.append(trial('b_svd_w0.1', 'gate_b', dict(base, hyperbolic_entailment_weight=.1,
                     hyperbolic_entailment_parent='svd_learned_radius'), base_trial=winner['name']))
    return out


def add_pmrl_screen(trials, suite, args):
    return [trial(f'c_pmrl_global_tau1_{tau1}', 'pmrl_screen', dict(PMRL_GLOBAL, tau1=tau1), steps=SCREEN)
            for tau1 in (.05, .07, .1)]


def add_main(trials, suite, args):
    """Three seeds for Ours (final), the kernel reference and matched PMRL."""
    ours = final_method(suite, trials)
    pmrl_screen = [t for t in trials if t['phase'] == 'pmrl_screen']
    pmrl = rank_by_holdout(suite, pmrl_screen)
    if not pmrl:
        sys.exit('run the pmrl_screen trials first')
    pmrl_model = pmrl[0][0]['model']
    kernel = next(t for t in trials if t['name'] == 'a_kernel_e0')
    out = [trial('main_pmrl_global_s50', 'main', pmrl_model)]
    for source, prefix in ((ours, 'main_ours'), (kernel, 'main_kernel')):
        out += [trial(f'{prefix}_s{seed}', 'main', source['model'], seed=seed, source=source['name'])
                for seed in EXTRA_SEEDS]
    out += [trial(f'main_pmrl_global_s{seed}', 'main', pmrl_model, seed=seed) for seed in EXTRA_SEEDS]
    return out


def add_lowdim(trials, suite, args):
    """Width sweep with tau re-screened per width (basis from utils.lowdim_parent)."""
    if not args.basis:
        sys.exit('--basis is required (pca/basis.pt from the 20260929 low-dim suite)')
    ours = final_method(suite, trials)['model']
    arms = dict(ours=ours, kernel=KERNEL, allpair=dict(LORENTZ, hyperbolic_score='allpair'))
    out = []
    for width, (arm, model) in itertools.product((64, 16), arms.items()):
        key = 'hyperbolic_spectral_temperature' if model['model_type'] == 'hyperbolic_spectral_pmrl' else 'hyperbolic_temperature'
        for tau in TEMPERATURES:
            out.append(trial(f'w{width}_{arm}_t{tau}', 'lowdim_screen', dict(model, contra_dim=width, **{key: tau}),
                             steps=SCREEN, run=dict(lowdim_init_basis=str(Path(args.basis).resolve()))))
    return out


TRIAL_SETS = dict(gate_a=add_gate_a, gate_a_promote=add_gate_a_promote, gate_a_contrast=add_gate_a_contrast,
                  gate_b=add_gate_b, pmrl_screen=add_pmrl_screen, main=add_main, lowdim=add_lowdim)


def cmd_add(args):
    suite = suite_dir(args)
    trials = load_trials(suite)
    new = TRIAL_SETS[args.set](trials, suite, args)
    existing = {t['name'] for t in trials}
    added = [t for t in new if t['name'] not in existing]
    save_trials(suite, trials + added)
    for t in added:
        print(f"added {t['name']} ({t['steps']} steps)")
    if len(added) < len(new):
        print(f'{len(new) - len(added)} already present')


# ---------------------------------------------------------------- configs

def holdout_entries(protocol, task):
    holdout = protocol['holdout']
    if not holdout:
        raise RuntimeError('suite has no holdout split')
    return [dict(type='annoindexed', training=False, name='finetune_area', txt=holdout['holdout_txt'],
                 vision=holdout['vision'], audio=holdout['audio'], vision_format='video_rawvideo',
                 vision_sample_num=holdout.get('vision_sample_num', 8), audio_sample_num=1,
                 task=task, n_workers=2, batch_size=64, strict_media=True,
                 loader_timeout=600, bounded_video_decode=True, video_decode_timeout=10,
                 drop_subtitles=True)]


def build_config(suite, t, stage):
    protocol = read_json(suite / 'protocol.json')
    source = suite / 'source_snapshot'
    template = {'train': 'train', 'holdout_audio': 'audio_evaluation', 'audio': 'audio_evaluation'}.get(stage, 'evaluation')
    cfg = copy.deepcopy(read_json(suite / 'templates' / f'{template}.json'))
    output = stage_dir(suite, t['name'], stage)
    run = cfg['run_cfg']
    run.update(default=str(source / 'config/pmrl/default_run_cfg.json'), output_dir=str(output),
               log_name=f"{t['name']}_{stage}", seed=t['seed'], resume=False, first_eval=False,
               log_steps=10, resource_profile=True)
    resources = protocol.get('resources', {})
    run['gpu_memory_limit_gib'] = resources.get('train_memory_gib' if stage == 'train' else 'eval_memory_gib', 30)
    run['checkpoint_lock'] = str(suite / 'checkpoint.lock')
    model = copy.deepcopy(read_json(suite / 'templates' / 'train.json')['model_cfg'])
    model.update(t['model'])
    model['default'] = str(source / 'config/pmrl/default_model_cfg.json')
    if stage == 'train':
        run.update(num_train_steps=FULL, stop_after_updates=t['steps'], new_lr=t['new_lr'],
                   new_params_name=['hyperbolic_'], valid_freq=1, final_checkpoint_only=True,
                   save_optimizer=resources.get('save_optimizer', False), save_best=False, checkpoint='', mode='training',
                   pretrain_dir=protocol['pretrain_dir'], amp_init_scale=32768.)
        run.update(t['run'])
        train = cfg['data_cfg']['train'][0]
        train.update(drop_subtitles=True)
        if protocol['holdout']:
            train['txt'] = protocol['holdout']['train_txt']
        cfg['data_cfg']['val'] = []
    else:
        run.update(mode='testing', checkpoint=str(trained_checkpoint(suite, t)), pretrain_dir='',
                   save_optimizer=False, stop_after_updates=0, lowdim_init_basis='')
        model.update(checkpointing=False, ret_bidirection_evaluation=True, itm_rerank_num=50,
                     evaluation_type='evaluation_mm',
                     ret_stage1_only=stage in ('audio', 'holdout_audio'),
                     ret_tv_only_candidates=stage in ('video', 'holdout_video'),
                     ret_native_tuple_candidates=stage == 'native',
                     ret_skip_euclidean_diagnostics=model['model_type'] == 'lorentz_omnimodal')
        if stage == 'holdout_video':
            cfg['data_cfg']['val'] = holdout_entries(protocol, 'ret%tva')
        elif stage == 'holdout_audio':
            cfg['data_cfg']['val'] = holdout_entries(protocol, 'ret%ta')
        elif t.get('smoke'):
            cfg['data_cfg']['val'] = holdout_entries(protocol, 'ret%tva')
        for entry in cfg['data_cfg']['val']:
            if stage.startswith('holdout') or t.get('smoke'):
                overrides = read_json(suite / 'templates' / 'train.json')['data_cfg']['train'][0].get('video_overrides')
                if overrides:
                    entry['video_overrides'] = overrides
    for entry in cfg['data_cfg'].get('train', []) + cfg['data_cfg'].get('val', []):
        entry['sample_trace_dir'] = str(Path(resources.get('logs_dir', REPO / '.tmp' / suite.name))
                                       / t['name'] / stage / 'media')
    cfg['model_cfg'] = model
    path = suite / 'trials' / t['name'] / 'configs' / f'{stage}.json'
    write_json(path, cfg)
    return path, output


def trained_checkpoint(suite, t):
    ckpts = sorted((stage_dir(suite, t['name'], 'train') / 'ckpt').glob('model_step_*.pt'),
                   key=lambda p: int(p.stem.rsplit('_', 1)[1]))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint for {t['name']}")
    step = int(ckpts[-1].stem.rsplit('_', 1)[1])
    if step != t['steps']:
        raise RuntimeError(f"{t['name']}: checkpoint step {step} != {t['steps']}")
    return ckpts[-1]


# ---------------------------------------------------------------- run

def gpu_free_gib(gpus):
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                     '--format=csv,noheader,nounits'], text=True).splitlines()
    values = {int(line.split(',')[0]): float(line.split(',')[1]) / 1024 for line in lines}
    return {int(gpu): values[int(gpu)] for gpu in gpus}


def run_bounded(command, suite, output, log_path, env, gpus, cap, port):
    resources = read_json(suite / 'protocol.json').get('resources', {})
    headroom = resources.get('gpu_headroom_gib', 8)
    while min(gpu_free_gib(gpus).values()) < cap + headroom + 1:
        print('Waiting for GPU memory:', gpu_free_gib(gpus), flush=True)
        time.sleep(30)
    if shutil.disk_usage(suite).free < (resources.get('disk_reserve_gib', 40) + 18) * 2**30:
        raise RuntimeError('Disk reserve reached; stopping before another stage')
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('127.0.0.1', port))
        except OSError:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
            command[command.index('--master_port') + 1] = str(port)
            print(f'Using available distributed port {port}', flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a') as log:
        process = subprocess.Popen(command, cwd=suite / 'source_snapshot', env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(output / 'execution.json', dict(pid=process.pid, command=command,
                   gpus=gpus, port=port, log=str(log_path), cap_gib=cap, started=time.time()))
        try:
            while process.poll() is None:
                free = gpu_free_gib(gpus)
                write_json(output / 'resource_status.json', dict(time=time.time(), free_gib=free))
                if min(free.values()) < headroom:
                    raise RuntimeError(f'GPU headroom below {headroom} GiB: {free}')
                time.sleep(5)
        except BaseException:
            # This process group was created above and contains only this stage.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    return process.returncode

def cmd_run(args):
    suite = suite_dir(args)
    trials = load_trials(suite)
    selected = [t for t in trials if (not args.names or t['name'] in args.names)
                and (not args.phase or t['phase'] in args.phase)]
    gpus = args.gpus.split(',')
    for t in selected:
        for stage in stages_for(t):
            if done(suite, t['name'], stage):
                continue
            failed = stage_dir(suite, t['name'], stage) / 'FAILED.json'
            if failed.exists() and not args.retry_failed:
                print(f"skip {t['name']}/{stage}: failed earlier (use --retry-failed)")
                if stage == 'train':
                    break
                continue
            if stage != 'train' and not done(suite, t['name'], 'train'):
                break
            path, output = build_config(suite, t, stage)
            command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes', '1', '--nproc_per_node', str(len(gpus)),
                       '--master_port', str(args.port), 'run.py', '--config', str(path)]
            print(f"[{time.strftime('%H:%M:%S')}] {t['name']}/{stage}: {' '.join(command)}", flush=True)
            if args.dry_run:
                continue
            output.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
            started = time.time()
            protocol = read_json(suite / 'protocol.json')
            logs = Path(protocol.get('resources', {}).get('logs_dir', REPO / '.tmp' / suite.name))
            log_path = logs / t['name'] / stage / 'stdout.log'
            (log_path.parent / 'log').mkdir(parents=True, exist_ok=True)
            if not (output / 'log').exists():
                (output / 'log').symlink_to(log_path.parent / 'log', target_is_directory=True)
            if not (output / 'stdout.log').exists():
                (output / 'stdout.log').symlink_to(log_path)
            try:
                code = run_bounded(command, suite, output, log_path, env, gpus,
                                   read_json(path)['run_cfg']['gpu_memory_limit_gib'], args.port)
            except Exception as error:
                write_json(output / 'FAILED.json', dict(returncode=1, error=repr(error), config=str(path)))
                raise
            record = dict(returncode=code, seconds=round(time.time() - started), config=str(path))
            if code == 0 and stage != 'train' and not (output / 'metrics.json').exists():
                code, record['error'] = 1, 'metrics.json missing'
            if code == 0 and stage == 'train':
                try:
                    trained_checkpoint(suite, t)
                except (FileNotFoundError, RuntimeError) as error:
                    code, record['error'] = 1, str(error)
            record['returncode'] = code
            write_json(output / ('DONE.json' if code == 0 else 'FAILED.json'), record)
            if code != 0:
                print(f"  FAILED (see {output / 'stdout.log'})")
                if args.stop_on_failure:
                    raise SystemExit(f"stopped after {t['name']}/{stage} failed")
                if stage == 'train':
                    break


# ---------------------------------------------------------------- metrics

def holdout_metrics(suite, name):
    video = stage_dir(suite, name, 'holdout_video') / 'metrics.json'
    audio = stage_dir(suite, name, 'holdout_audio') / 'metrics.json'
    if not (video.exists() and audio.exists() and done(suite, name, 'holdout_video')
            and done(suite, name, 'holdout_audio')):
        return None
    result = {}
    for stage, values in read_json(video).items():
        for key, value in values.items():
            for kind in ('itc', 'itm'):
                if f'_ret_{kind}_' in key:
                    result[f'{kind}_r1'] = (value['video_r1'] + value['txt_r1']) / 2
    for values in read_json(audio).values():
        for key, value in values.items():
            if '_ret_itc_' in key:
                result['audio_r1'] = (value['video_r1'] + value['txt_r1']) / 2
    return result if {'itc_r1', 'itm_r1', 'audio_r1'} <= set(result) else None


def test_metrics(suite, name):
    result = {}
    video = stage_dir(suite, name, 'video') / 'metrics.json'
    audio = stage_dir(suite, name, 'audio') / 'metrics.json'
    native = stage_dir(suite, name, 'native') / 'metrics.json'
    if video.exists():
        v = load_video_metrics(video)
        result.update(stage1=v['itc_macro_r1'], top50=v['itm_macro_r1'])
    if audio.exists():
        result['audio'] = load_audio_metrics(audio)['macro_r1']
    if native.exists():
        n = load_video_metrics(native)
        result.update(native_stage1=n['itc_macro_r1'], native_top50=n['itm_macro_r1'])
    return result


def train_geometry(suite, name, last=50):
    path = stage_dir(suite, name, 'train') / 'metrics.jsonl'
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r['geometry'] for r in rows[-last:] if 'geometry' in r]
    keys = {k for r in rows for k, v in r.items() if isinstance(v, (int, float))}
    return {k: sum(r[k] for r in rows if k in r) / sum(k in r for r in rows) for k in sorted(keys)}


def selection_key(h):
    return (h['itm_r1'], h['itc_r1'], h['audio_r1'])


def rank_by_holdout(suite, trials):
    scored = [(t, holdout_metrics(suite, t['name'])) for t in trials]
    scored = [(t, selection_key(h)) for t, h in scored if h]
    return sorted(scored, key=lambda x: x[1], reverse=True)


def best_full(suite, trials, predicate):
    ranked = rank_by_holdout(suite, [t for t in trials if is_full(t) and t['seed'] == 50 and predicate(t)])
    if not ranked:
        sys.exit('no finished full trial matches; run the earlier set first')
    return ranked[0][0]


def gate_winner(suite):
    path = suite / 'gate_a.json'
    if not path.exists():
        sys.exit('run gate-a first')
    gate = read_json(path)
    trials = {t['name']: t for t in load_trials(suite)}
    return trials[gate['winner']]


def final_method(suite, trials):
    path = suite / 'gate_b.json'
    if not path.exists():
        sys.exit('run gate-b first')
    gate = read_json(path)
    return {t['name']: t for t in trials}[gate['final']]


# ---------------------------------------------------------------- gates

def cmd_gate_a(args):
    suite = suite_dir(args)
    trials = load_trials(suite)
    kernel = next(t for t in trials if t['name'] == 'a_kernel_e0')
    candidate = best_full(suite, trials, lambda t: t['phase'] == 'gate_a'
                          and t['model']['model_type'] == 'lorentz_omnimodal')
    # Decide on the holdout; test deltas are reported, never used to decide.
    hk, hm = holdout_view(suite, kernel['name']), holdout_view(suite, candidate['name'])
    k, m = test_metrics(suite, kernel['name']), test_metrics(suite, candidate['name'])
    if not (hk and hm):
        sys.exit('missing holdout metrics')
    delta = {x: hm[x] - hk[x] for x in GATE_A}
    passed = all(delta[x] >= GATE_A[x] for x in GATE_A)
    result = dict(passed=passed, candidate=candidate['name'], reference=kernel['name'],
                  selected_by='holdout itm, itc, audio R@1', decided_on='holdout',
                  holdout_delta=delta, thresholds=GATE_A, candidate_test=m, reference_test=k,
                  test_delta={x: m[x] - k[x] for x in GATE_A if x in m and x in k},
                  winner=candidate['name'] if passed else kernel['name'],
                  note=None if passed else 'Fallback: kernel score; Perron pair kept as analysis and proxy.')
    write_json(suite / 'gate_a.json', result)
    print(json.dumps(result, indent=2))


def holdout_view(suite, name):
    h = holdout_metrics(suite, name)
    return h and dict(stage1=h['itc_r1'], top50=h['itm_r1'], audio=h['audio_r1'])


def cmd_gate_b(args):
    suite = suite_dir(args)
    trials = load_trials(suite)
    winner = gate_winner(suite)
    base = holdout_view(suite, winner['name'])
    if not base:
        sys.exit('missing holdout metrics for the Gate-A winner')
    hierarchy = read_json(args.hierarchy) if args.hierarchy else {}
    rows, keep = [], []
    for t in [x for x in trials if x['phase'] == 'gate_b']:
        held, geometry = holdout_view(suite, t['name']), train_geometry(suite, t['name'])
        cones = [v for k, v in geometry.items() if k.startswith('cone_satisfied_')]
        cone = sum(cones) / len(cones) if cones else None
        row = dict(trial=t['name'], holdout_top50=held and held['top50'], cone_satisfied=cone,
                   delta_top50=held and held['top50'] - base['top50'], hierarchy_gain=None,
                   test=test_metrics(suite, t['name']))
        if t['name'] in hierarchy and winner['name'] in hierarchy:
            row['hierarchy_gain'] = hierarchy[t['name']] - hierarchy[winner['name']]
        ok_cone = cone is not None and cone >= GATE_B['cone']
        ok_top50 = row['delta_top50'] is not None and row['delta_top50'] >= GATE_B['top50']
        ok_hier = row['hierarchy_gain'] is not None and row['hierarchy_gain'] > args.hierarchy_std
        row['passes'] = bool(ok_cone and ok_top50 and ok_hier)
        rows.append(row)
        if row['passes']:
            keep.append((selection_key(holdout_metrics(suite, t['name'])), t['name']))
    final = max(keep)[1] if keep else winner['name']
    result = dict(rows=rows, thresholds=dict(GATE_B, hierarchy_std=args.hierarchy_std),
                  hierarchy_source=args.hierarchy, final=final,
                  decision='keep cone entailment' if keep else
                  'drop the cone loss; state hierarchy via the Perron proxy and diagnostics')
    if not hierarchy:
        result['warning'] = 'no hierarchy metrics given (--hierarchy); no parent can pass'
    write_json(suite / 'gate_b.json', result)
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------- summaries

def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return cov / var if var else float('nan')


def cmd_rank_agreement(args):
    """Spearman rho between short-screen and full-run metrics.

    --pairs: JSON list of {name, screen, full} rows (e.g. hand-extracted from the
    20260929 suite's 12 screens / 3 promotions and the 0821 minimal grid);
    without it, uses this suite's promoted screens (screen_of links).
    """
    if args.pairs:
        rows = read_json(args.pairs)
    else:
        suite = suite_dir(args)
        rows = []
        for t in load_trials(suite):
            if 'screen_of' in t:
                s, f = holdout_metrics(suite, t['screen_of']), holdout_metrics(suite, t['name'])
                if s and f:
                    rows.append(dict(name=t['name'], screen=s['itm_r1'], full=f['itm_r1']))
    if len(rows) < 3:
        sys.exit(f'need >= 3 pairs, have {len(rows)}')
    rho = spearman([r['screen'] for r in rows], [r['full'] for r in rows])
    print(json.dumps(dict(n=len(rows), spearman=rho, trust_screens=rho >= .7, rows=rows), indent=2))


def cmd_summarize(args):
    suite = suite_dir(args)
    lines = ['| trial | phase | steps | seed | holdout itm/itc/audio | test Stage-1 | Top-50 | audio | native Top-50 | cone sat. | gap/k min | status |',
             '|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|']
    summary = []
    for t in load_trials(suite):
        h, test, g = holdout_metrics(suite, t['name']), test_metrics(suite, t['name']), train_geometry(suite, t['name'])
        status = ','.join(s for s in stages_for(t) if not done(suite, t['name'], s)) or 'complete'
        cones = [v for k, v in g.items() if k.startswith('cone_satisfied_')]
        row = dict(trial=t['name'], phase=t['phase'], holdout=h, test=test, geometry=g, pending=status)
        summary.append(row)
        f = lambda x: '' if x is None else f'{x:.2f}'
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            t['name'], t['phase'], t['steps'], t['seed'],
            '' if not h else f"{h['itm_r1']:.2f}/{h['itc_r1']:.2f}/{h['audio_r1']:.2f}",
            f(test.get('stage1')), f(test.get('top50')), f(test.get('audio')), f(test.get('native_top50')),
            f(sum(cones) / len(cones) if cones else None), f(g.get('perron_gap_over_k_min')), status))
    write_json(suite / 'summary.json', summary)
    (suite / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--suite')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('init')
    p.add_argument('--templates', required=True)
    p.add_argument('--pretrain-dir', required=True)
    p.add_argument('--holdout-dir')
    p.add_argument('--weights-dir')
    p.add_argument('--logs-dir')
    p.add_argument('--train-memory-gib', type=float, default=28)
    p.add_argument('--eval-memory-gib', type=float, default=27)
    p = sub.add_parser('add')
    p.add_argument('set', choices=sorted(TRIAL_SETS))
    p.add_argument('--top', type=int, default=2)
    p.add_argument('--basis')
    p = sub.add_parser('run')
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--port', type=int, default=29650)
    p.add_argument('--names', nargs='*')
    p.add_argument('--phase', nargs='*')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--stop-on-failure', action='store_true')
    sub.add_parser('gate-a')
    p = sub.add_parser('gate-b')
    p.add_argument('--hierarchy', help='JSON {trial: hierarchy metric}, e.g. ActivityNet ancestor F1')
    p.add_argument('--hierarchy-std', type=float, default=.3, help='seed std the gain must exceed')
    p = sub.add_parser('rank-agreement')
    p.add_argument('--pairs')
    sub.add_parser('summarize')
    args = parser.parse_args()
    if args.command == 'init':
        if not args.suite:
            sys.exit('init needs --suite')
        return cmd_init(args)
    dict(add=cmd_add, run=cmd_run, summarize=cmd_summarize, **{
        'gate-a': cmd_gate_a, 'gate-b': cmd_gate_b, 'rank-agreement': cmd_rank_agreement})[args.command](args)


if __name__ == '__main__':
    main()
