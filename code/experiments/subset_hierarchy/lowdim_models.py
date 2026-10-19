"""Pretrained model adapters and paired geometry-neutral head initialization."""
import fcntl
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import types
import torch
from torch import nn
from easydict import EasyDict as edict
from experiments.subset_hierarchy.lowdim_common import *
from utils.lowdim_geometry import project_native_state,initialize_subset_heads


def install_checkpoint_policy():
    """Process-local policy, also covering author modules with imported aliases."""
    import torch.utils.checkpoint as cp
    original=getattr(cp.checkpoint, '_lowdim_original', cp.checkpoint)
    def checkpoint(function,*args,**kwargs):
        kwargs['use_reentrant']=False
        return original(function,*args,**kwargs)
    checkpoint._lowdim_original=original
    cp.checkpoint=checkpoint
    for name,module in list(sys.modules.items()):
        if name.startswith(('lowdim_author_', 'model.')) and hasattr(module,'checkpoint'):
            if callable(module.checkpoint):module.checkpoint=checkpoint


def native_class(name,roots,p):
    if name=='pmrl':
        from model.pmrl import PMRL
        return PMRL
    source=resolve_ref(p['author_sources'][name],roots)
    if name=='gram':
        spec=importlib.util.spec_from_file_location('utils.volume',source/'utils/volume.py')
        module=importlib.util.module_from_spec(spec);sys.modules['utils.volume']=module;spec.loader.exec_module(module)
    package_name='lowdim_author_'+name
    package=types.ModuleType(package_name);package.__path__=[str(source/'model')];sys.modules[package_name]=package
    return getattr(importlib.import_module(package_name+'.'+name),name.upper())


def stage_checkpoint(roots,p,name):
    source=resolve_ref(p['checkpoints'][name],roots);expected=p['checkpoint_sha256'][name]
    folder=roots['workspace']/'.cache/lowdim_checkpoints';folder.mkdir(parents=True,exist_ok=True)
    target=folder/(expected+'.pt');marker=folder/(expected+'.json')
    with (folder/(expected+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if target.exists() and marker.exists() and read_json(marker).get('size')==target.stat().st_size:return target
        if shutil.disk_usage(folder).free<source.stat().st_size+20*2**30:return source
        tmp=target.with_suffix(f'.{os.getpid()}.partial');shutil.copyfile(source,tmp)
        if file_hash(tmp)!=expected:raise RuntimeError('Staged checkpoint hash mismatch')
        os.replace(tmp,target);write_json(marker,dict(sha256=expected,size=target.stat().st_size))
    return target


def basis_for(roots,dataset,name,width):
    if width==512:return torch.eye(512),dict(kind='identity',width=512)
    path=roots['suite']/'bases'/dataset/(name+'.pt')
    pack=torch.load(path,map_location='cpu')
    ready=read_json(roots['suite']/'data'/dataset/'READY.json')
    p=read_json(roots['suite']/'protocol.json')
    if pack['train_manifest_sha256']!=ready['manifest_hashes']['train'] or pack['checkpoint_sha256']!=p['checkpoint_sha256'][name]:
        raise ValueError('PCA basis provenance mismatch')
    return pack['basis'],dict(kind='train_only_pca',resource=dict(root='suite',path=str(path.relative_to(roots['suite']))),
        sha256=file_hash(path),train_ids_sha256=pack['train_ids_sha256'])


def make_model(roots,dataset,method,width,device,*,initial_only=False,stage_local=True):
    p=read_json(roots['suite']/'protocol.json');family='vast' if ours(method) else method
    basis,basis_audit=basis_for(roots,dataset,family,width)
    settings=p['datasets'][dataset];cfg=dict(p['model_cfg'])
    if method=='gram':cfg.update(p['gram_hps']['model_cfg'])
    # The complete model configuration is embedded in the frozen protocol;
    # do not carry a source-host-only default-config filename into execution.
    cfg.pop('default',None)
    cfg.update(contra_dim=width,checkpointing=not initial_only,max_caption_len=settings['max_caption_len'],
        max_vision_sample_num=settings['vision_frames'],max_audio_sample_num=settings['audio_windows'],
        frozen_vision=False,frozen_audio=False,tau1=.5,tau2=.1,lambda_itm=.1,itm_ratio=.1,
        pmrl_global_negatives=True,subset_geometry=geometry_for(method) if ours(method) else 'hyperbolic')
    assets=resolve_ref(p['assets'],roots);work=roots['workspace']/'.cache/lowdim_assets';work.mkdir(parents=True,exist_ok=True)
    link=work/'pretrained_weights'
    if link.exists() or link.is_symlink():
        if link.resolve()!=assets:raise RuntimeError('Asset link points to a different root')
    else:
        try:link.symlink_to(assets,target_is_directory=True)
        except FileExistsError:
            if link.resolve()!=assets:raise RuntimeError('Concurrent asset link mismatch')
    if ours(method):
        from model.subset_hierarchy import SubsetHierarchy
        cls=SubsetHierarchy
    else:cls=native_class(method,roots,p)
    previous=Path.cwd()
    original_load=torch.load
    def load_on_cpu(*args,**kwargs):
        kwargs.setdefault('map_location','cpu')
        return original_load(*args,**kwargs)
    try:
        os.chdir(work);torch.load=load_on_cpu;model=cls(edict(cfg))
    finally:
        torch.load=original_load;os.chdir(previous)
    install_checkpoint_policy()
    checkpoint=stage_checkpoint(roots,p,family) if stage_local else resolve_ref(p['checkpoints'][family],roots)
    raw=torch.load(checkpoint,map_location='cpu');source_state=model.modify_checkpoint(raw.get('model',raw))
    state,projected=project_native_state(source_state,basis,width)
    missing,unexpected=model.load_state_dict(state,strict=False)
    allowed_missing={'vision_encoder.logit_scale'}|({'contra_head_d.linear.weight'} if method=='gram' else set())
    illegal=[k for k in missing if k not in allowed_missing and not (ours(method) and k.startswith('subset_heads.'))]
    if illegal or set(unexpected)-{'vision_encoder.text.logit_scale'}:raise RuntimeError(f'Checkpoint load mismatch: {illegal}, {unexpected}')
    if ours(method):model.subset_heads=initialize_subset_heads(p['input_dims'],source_state,basis,width,geometry_for(method))
    del raw,source_state,state
    model.requires_grad_(False)
    if not initial_only:
        for module in (model.vision_encoder.visual,model.audio_encoder,model.multimodal_encoder.bert):module.requires_grad_(True)
        if ours(method):model.subset_heads.requires_grad_(True)
        else:
            for attr in ['contra_head_t','contra_head_v','contra_head_a','contra_head_va','itm_head','hidden_trans_vision_multimodal','hidden_trans_audio_multimodal']:
                getattr(model,attr).requires_grad_(True)
            for attr in ['contra_temp','vision_frame_embedding','audio_frame_embedding','vision_type_embeddings','audio_type_embeddings']:
                getattr(model,attr).requires_grad_(True)
        for module in model.modules():module.subset_non_reentrant=True
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        is_head=name.startswith('subset_heads.') if ours(method) else name.startswith('contra_head_')
        if is_head:
            h.update(name.encode());h.update(value.detach().float().contiguous().numpy().tobytes())
    audit=dict(family=family,dimension=width,checkpoint_sha256=p['checkpoint_sha256'][family],basis=basis_audit,
        heads_sha256=h.hexdigest(),missing=list(missing),unexpected=list(unexpected),projected_heads=projected,
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),model_cfg=cfg)
    return model.to(device),audit


class TrainingAdapter(nn.Module):
    def __init__(self,model,method):super().__init__();self.net=model;self.method=method
    def forward(self,batch):
        if ours(self.method):return self.net(batch)
        return self.net.forward_ret(batch,'ret%tv%ta',compute_loss=True)
