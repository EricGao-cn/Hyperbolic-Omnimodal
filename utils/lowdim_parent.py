"""Matched native-width parent ablations and train-only PCA initialization."""
import copy
from pathlib import Path
import torch
from utils.omnimodal_experiments import read_json, write_json

ARMS = ('no_ent', 'text', 'mean', 'svd')

def project_checkpoint(state, basis, width):
    if basis.shape != (512, 512) or width not in (16, 64):
        raise ValueError('Expected a 512D basis and width 16 or 64')
    projection = basis[:, :width].T.float().contiguous()
    torch.testing.assert_close(projection @ projection.T, torch.eye(width), atol=2e-5, rtol=2e-5)
    result = dict(state)
    changed = []
    for key, value in state.items():
        if key.startswith('contra_head_'):
            if value.shape[0] != 512 or value.ndim not in (1, 2):
                raise ValueError(f'Unexpected contrast head: {key} {value.shape}')
            result[key] = projection @ value.float()
            changed.append(key)
    if not all(f'contra_head_{m}.linear.weight' in changed for m in 'tva'):
        raise ValueError('VAST T/V/A contrast heads missing')
    return result

def validate_initializer_keys(missing, unexpected):
    # The existing VAST loader leaves this unused vision temperature at its
    # encoder default and ignores the released checkpoint's text-side key.
    allowed_missing = {'vision_encoder.logit_scale'}
    allowed_unexpected = {'vision_encoder.text.logit_scale'}
    illegal = [k for k in missing if not k.startswith(('hyperbolic_', '_hyperbolic_')) and k not in allowed_missing]
    extra = set(unexpected) - allowed_unexpected
    if illegal or extra:
        raise RuntimeError(f'VAST initialization mismatch: missing={illegal}, unexpected={sorted(extra)}')
    return dict(missing=list(missing), unexpected=list(unexpected),
                note='Explicitly audited legacy VAST vision/text temperature-key mismatch; same policy as historical initializer')

def fit_basis(features):
    moment = torch.zeros(512, 512, dtype=torch.float64)
    for m in 'tva':
        x = torch.nn.functional.normalize(features[m].double(), dim=-1)
        if x.ndim != 2 or x.shape[1] != 512 or not torch.isfinite(x).all():
            raise ValueError('Invalid PCA features')
        moment += x.T @ x / (3 * len(x))
    values, basis = torch.linalg.eigh(moment)
    values, basis = values.flip(0), basis.flip(1)
    pivot = basis.abs().argmax(0)
    basis *= torch.where(basis[pivot, torch.arange(512)] >= 0, 1., -1.)
    return basis.float(), values

def specs(smoke=False):
    return [dict(name=f'd{d}_{arm}' + ('_smoke' if smoke else ''), width=d, arm=arm,
                 seed=50, phase='smoke' if smoke else 'full', steps=3 if smoke else 1360,
                 model_type='hyperbolic_spectral_pmrl') for d in (64, 16) for arm in ARMS]

def stages_for(trial):
    if trial['phase'] == 'smoke':
        return ['train', 'evaluation', 'audio_evaluation', 'geometry', 'msrvtt', 'audiocaps_audioset', 'activitynet']
    return ['train', 'evaluation', 'audio_evaluation', 'initial_geometry', 'geometry',
            'msrvtt', 'audiocaps_audioset', 'activitynet']

def model_overrides(trial):
    arm = trial['arm']
    return dict(model_type='hyperbolic_spectral_pmrl', contra_dim=trial['width'],
                hyperbolic_spectral_temperature=.03, hyperbolic_spectral_anchor='text',
                hyperbolic_entailment_weight=0. if arm == 'no_ent' else .1,
                hyperbolic_entailment_edges=[['t','v'],['t','a']] if arm == 'text' else [],
                hyperbolic_entailment_parent={'no_ent':'modality_edges','text':'modality_edges',
                                             'mean':'tangent_mean','svd':'svd_learned_radius'}[arm],
                hyperbolic_parent_contraction=1., hyperbolic_parent_floor_margin=.001,
                hyperbolic_entailment_min_radius=.1, lambda_itm=.1)

def config_for(suite, trial, stage, attempt, memory_cap):
    suite = Path(suite); protocol = read_json(suite/'protocol.json')
    root = suite/'trials'/trial['name']; output = root/stage/f'attempt_{attempt:02d}'
    logs = Path(protocol['logs'])/trial['name']/stage/f'attempt_{attempt:02d}'
    source = suite/'source_snapshot'
    key = stage if stage in ('train','evaluation','audio_evaluation') else 'train'
    cfg = copy.deepcopy(read_json(suite/'templates'/f'{key}.json'))
    cfg['model_cfg'].update(model_overrides(trial))
    cfg['model_cfg']['default'] = str(source/'config/pmrl/default_model_cfg.json')
    run = cfg['run_cfg']
    run.update(default=str(source/'config/pmrl/default_run_cfg.json'), output_dir=str(output),
               seed=50, log_name=trial['name']+'_'+stage, resource_profile=True,
               gpu_memory_limit_gib=memory_cap, checkpoint_lock=protocol['checkpoint_lock'],
               resume=False, first_eval=False, save_best=False, log_steps=10,
               num_train_steps=1360, gradient_accumulation_steps=1)
    if stage == 'train':
        run.update(pretrain_dir=protocol['pretrain_dir'], checkpoint='', mode='training',
                   lowdim_init_basis=str(suite/'pca/basis.pt'), final_checkpoint_only=True,
                   stop_after_updates=trial['steps'], save_optimizer=trial['phase']=='full',
                   new_lr=1e-4, new_params_name=['hyperbolic_'], amp_init_scale=32768.)
        cfg['data_cfg']['train'][0].update(batch_size=64, n_workers=4, drop_subtitles=True)
    else:
        trained = read_json(root/'train_complete.json')
        run.update(mode='testing', checkpoint=trained['checkpoint'], pretrain_dir='',
                   lowdim_init_basis='', save_optimizer=False, stop_after_updates=0)
        cfg['model_cfg'].update(checkpointing=False, ret_stage1_only=stage=='audio_evaluation',
                                itm_rerank_num=50)
    for split in ('train','val'):
        for data in cfg['data_cfg'][split]:
            data['sample_trace_dir'] = str(logs/'media_trace')
            if split=='val' and trial['phase']=='smoke':
                data['txt'] = str(suite/'smoke_manifests'/f"{data['name']}.json")
    output.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    (output/'log').symlink_to(logs)
    path = root/'configs'/f'{stage}_{attempt:02d}.json'; write_json(path,cfg)
    return path, output, logs
