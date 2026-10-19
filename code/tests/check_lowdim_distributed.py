"""CPU/Gloo equivalence of paired subset losses and native baseline objectives.

Native encoders are replaced by differentiable toy feature/conditioning
fixtures. The authors' forward_ret code (including ITM/DAM) is executed with
fixed negative IDs. Full-encoder CUDA checks remain a separate required gate.
"""
import argparse
from contextlib import contextmanager,ExitStack
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from easydict import EasyDict as edict
from utils.subset_hierarchy import SubsetHeads
from utils.lowdim_geometry import loss
from experiments.subset_hierarchy.lowdim_models import native_class
from experiments.subset_hierarchy.lowdim_common import read_json,DEFAULT_SUITE


class TinyBert(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(32,6);self.context=nn.Linear(6,6,bias=False)
    def forward(self,input_ids,attention_mask,encoder_hidden_states):
        return edict(last_hidden_state=self.embedding(input_ids)+self.context(encoder_hidden_states.mean(1))[:,None,:])


class TinyITM(nn.Linear):
    def forward(self,x):return super().forward(x.float())


class NativeFixture(nn.Module):
    def __init__(self,name,cls,width):
        super().__init__();self.name=name;self.cls=cls
        self.transforms=nn.ModuleDict({m:nn.Linear(7,width,bias=False) for m in 'tva'})
        self.contexts=nn.ModuleDict({m:nn.Linear(7,6,bias=False) for m in 'va'})
        self.multimodal_encoder=nn.Module();self.multimodal_encoder.bert=TinyBert();self.itm_head=TinyITM(6,2)
        self.config=edict(pmrl_global_negatives=True);self.ret_pairwise_stage1_train=False
        self.contra_temp=nn.Parameter(torch.tensor(.07));self.tau1=.5;self.tau2=.1;self.lambda_itm=.1;self.itm_ratio=.1
    def batch_get(self,batch,key):
        if key in batch:return batch[key]
        if key.startswith('feat_'):
            s=key[5:];value=torch.stack([self.transforms[m](batch.x[m]) for m in s]).mean(0)
            return F.normalize(value,dim=-1)
        if key.startswith('condition_feats_'):
            s=key[len('condition_feats_'):]
            return torch.stack([self.contexts[m](batch.x[m]) for m in s],1)
        raise KeyError(key)
    @staticmethod
    def _distributed_pairwise_batch(values):
        from model.pmrl import PMRL
        return PMRL._distributed_pairwise_batch(values)
    def forward(self,batch):return self.cls.forward_ret(self,batch,'ret%tv%ta',True)


@contextmanager
def fixed_negatives(offset,batch_size,total):
    count=0
    def draw(weights,num_samples,*args,**kwargs):
        nonlocal count
        i=offset+count%batch_size;count+=1
        return torch.tensor([(i+1)%total],device=weights.device)
    with patch('torch.multinomial',draw),patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self):yield


@contextmanager
def single_rank_native(cls):
    gl=cls.forward_ret.__globals__
    from model.pmrl import PMRL
    helpers=PMRL._distributed_pairwise_batch.__globals__
    with ExitStack() as stack:
        stack.enter_context(patch.dict(gl,{'concat_all_gather':lambda x:x.detach().clone(),'all_gather_with_grad':lambda x:x}))
        if helpers is not gl:stack.enter_context(patch.dict(helpers,{'all_gather_with_grad':lambda x:x}))
        stack.enter_context(patch.object(dist,'get_rank',lambda:0));stack.enter_context(patch.object(dist,'get_world_size',lambda:1))
        yield


def batch(x,start,stop):
    ids=torch.arange(start,stop)
    return edict(raw_captions=['fixture']*(stop-start),x={m:v[start:stop] for m,v in x.items()},
        caption_tokens=edict(input_ids=torch.stack([ids+1,ids+2,ids+3],1),attention_mask=torch.ones(stop-start,3,dtype=torch.long)))


def check_equal(local,dense,expected_loss,actual_loss,rank,label,tolerance=5e-4):
    value=actual_loss.detach().double().clone();dist.all_reduce(value);value/=2
    torch.testing.assert_close(value,expected_loss.detach().double(),rtol=1e-5,atol=2e-6,msg=label+' loss')
    for (name,p),(ref_name,q) in zip(local.named_parameters(),dense.named_parameters()):
        assert name==ref_name
        if p.grad is None or q.grad is None:
            assert p.grad is None and q.grad is None,(label,name)
        else:torch.testing.assert_close(p.grad,q.grad,rtol=tolerance,atol=2e-5,msg=label+' '+name)
    if rank==0:print('PASS',label,flush=True)


def worker(rank,rendezvous,suite,author_root,data_root):
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    for geometry in ['hyperbolic','euclidean']:
        for weight in [0.,.1]:
            torch.manual_seed(50);dense=SubsetHeads({'t':6,'v':8,'a':5},8,geometry)
            inputs={m:torch.randn(8,d) for m,d in dense.input_dims.items()}
            local=copy.deepcopy(dense);expected,_=loss(dense(inputs),geometry,weight,distributed=False);expected.backward()
            wrapped=DDP(local)
            actual,_=loss(wrapped({m:v[rank*4:(rank+1)*4] for m,v in inputs.items()}),geometry,weight)
            actual.backward();check_equal(local,dense,expected,actual,rank,f'{geometry} weight={weight}',tolerance=1e-3)
    roots=dict(workspace=ROOT,data=Path(data_root),suite=Path(suite),authors=Path(author_root))
    protocol=read_json(Path(suite)/'protocol.json')
    for name in ['vast','gram','pmrl']:
        cls=native_class(name,roots,protocol)
        for width in [8,512]:
            torch.manual_seed(60+width);dense=NativeFixture(name,cls,width);local=copy.deepcopy(dense)
            inputs={m:torch.randn(8,7) for m in 'tva'}
            with single_rank_native(cls),fixed_negatives(0,8,8):expected=sum(dense(batch(inputs,0,8)).values())
            expected.backward()
            wrapped=DDP(local,find_unused_parameters=True)
            with fixed_negatives(rank*4,4,8):actual=sum(wrapped(batch(inputs,rank*4,(rank+1)*4)).values())
            actual.backward();check_equal(local,dense,expected,actual,rank,f'native {name} D={width}',tolerance=2e-3)
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,default=DEFAULT_SUITE);p.add_argument('--author-root',type=Path,default=ROOT.parent/'main_result_baselines');p.add_argument('--data-root',type=Path,default=Path('/data-nfs/xiaoboxia/YuxuanGao'));args=p.parse_args()
    (ROOT/'.tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='lowdim_gloo_',dir=ROOT/'.tmp') as tmp:
        mp.spawn(worker,args=(str(Path(tmp)/'rdzv'),str(args.suite),str(args.author_root),str(args.data_root)),nprocs=2,join=True)
