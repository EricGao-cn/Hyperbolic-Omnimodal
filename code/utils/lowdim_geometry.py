"""Paired Euclidean/hyperbolic subset objectives; legacy objectives are unchanged."""
import math
import torch
from torch.nn import functional as F
from utils.subset_hierarchy import (SubsetHeads,SUBSETS,EDGES,MINIMAL_PAIRS,
    pair_distance,gather_candidates,cone_violation,R_MIN,R_MAX)
from utils.subset_mechanisms import cone_parts,cap_width


def euclidean_cone_parts(parent,child,k=.2):
    p,q=parent.double(),child.double();radius=p.norm(dim=-1);delta=q-p
    if (radius<=k).any():raise ValueError('Euclidean cone apex must have radius > K')
    aperture=torch.asin((k/radius).clamp(max=1-1e-12))
    cosine=(p*delta).sum(-1)/(radius*delta.norm(dim=-1)).clamp_min(1e-12)
    angle=torch.acos(cosine.clamp(-1+1e-12,1-1e-12))
    return torch.where(delta.square().sum(-1)<=1e-24,angle*0,angle),aperture


def cone_parts_for(parent,child,geometry):
    if geometry=='hyperbolic':return cone_parts(parent,child)
    if geometry=='euclidean':return euclidean_cone_parts(parent,child)
    raise ValueError(geometry)


def loss(z,geometry='hyperbolic',entailment_weight=.1,temperature=.07,distributed=True):
    if geometry not in ('hyperbolic','euclidean'):raise ValueError(geometry)
    candidates,offset=gather_candidates(z) if distributed else (z,0)
    targets=torch.arange(len(z['t']),device=z['t'].device)+offset
    terms=[]
    for a,b in MINIMAL_PAIRS:
        ab=-pair_distance(z[a],candidates[b],geometry)/temperature
        ba=-pair_distance(z[b],candidates[a],geometry)/temperature
        terms.append(.5*(F.cross_entropy(ab,targets)+F.cross_entropy(ba,targets)))
    contrastive=torch.stack(terms).mean()
    violations=[]
    for p,c in EDGES:
        phi,omega=cone_parts_for(z[p],z[c],geometry);violations.append((phi-omega).relu())
    entailment=torch.stack(violations).mean()
    total=contrastive+entailment_weight*entailment
    return total,dict(contrastive=contrastive.detach(),entailment=entailment.detach(),
        weighted_entailment=(entailment_weight*entailment).detach(),
        strict_edge_satisfaction=(torch.stack(violations)<=1e-7).double().mean().detach()*100,
        candidate_count=torch.tensor(len(candidates['t']),device=total.device))


def fit_basis(features):
    if set(features)!=set('tva'):raise ValueError('PCA needs T/V/A training features')
    moment=torch.zeros(512,512,dtype=torch.float64)
    for m in 'tva':
        x=features[m].double()
        if x.ndim!=2 or x.shape[1]!=512 or len(x)==0 or not torch.isfinite(x).all():raise ValueError('Invalid PCA features')
        x=F.normalize(x,dim=-1);moment+=x.T@x/(3*len(x))
    values,vectors=torch.linalg.eigh(moment);values=values.flip(0);vectors=vectors.flip(1)
    pivot=vectors.abs().argmax(dim=0)
    vectors*=torch.where(vectors[pivot,torch.arange(512)]>=0,1.,-1.)
    return vectors.float(),values


def projection(basis,width):
    if basis.shape!=(512,512) or width not in (8,32,128,512):raise ValueError('Unsupported basis/width')
    p=torch.eye(512,dtype=torch.float32) if width==512 else basis[:,:width].T.float().contiguous()
    torch.testing.assert_close(p@p.T,torch.eye(width),atol=2e-5,rtol=2e-5)
    return p


def project_native_state(state,basis,width):
    p=projection(basis,width);result=dict(state);changed=[]
    for key,value in state.items():
        if key.startswith('contra_head_'):
            if value.ndim not in (1,2) or value.shape[0]!=512:raise ValueError(f'Unexpected native head {key}: {value.shape}')
            result[key]=p@value.float();changed.append(key)
    if not all(f'contra_head_{m}.linear.weight' in changed for m in 'tva'):raise ValueError('Missing T/V/A heads')
    return result,changed


def initialize_subset_heads(input_dims,source_state,basis,width,geometry):
    # Initialize in 512D first, then apply one shared coordinate projection.
    high=SubsetHeads(input_dims,512,geometry);high.initialize_vast(source_state)
    low=SubsetHeads(input_dims,width,geometry);p=projection(basis,width)
    with torch.no_grad():
        for s in SUBSETS:
            low.direction[s].weight.copy_(p@high.direction[s].weight)
            low.direction[s].bias.copy_(p@high.direction[s].bias)
            low.radius[s].weight.zero_()
            low.radius[s].bias.fill_(math.log((1.-R_MIN)/(R_MAX-1.)))
    return low


@torch.no_grad()
def geometry_metrics(z,geometry):
    if set(z)!=set(SUBSETS):raise ValueError('Expected all seven subset heads')
    r={s:x.double().norm(dim=-1) for s,x in z.items()}
    if geometry=='hyperbolic':r={s:v.asinh() for s,v in r.items()}
    result={'radii':{s:dict(mean=float(v.mean()),std=float(v.std()),p05=float(v.quantile(.05)),p95=float(v.quantile(.95)),above_3p9_pct=float((v>3.9).double().mean()*100)) for s,v in r.items()},'edges':{}}
    successful=[];negative=[]
    for p,c in EDGES:
        phi,omega=cone_parts_for(z[p],z[c],geometry);violation=(phi-omega).relu()
        np_,nw=cone_parts_for(z[p],z[c].roll(1,0),geometry)
        ok=violation<=1e-7;bad=(np_-nw).relu()<=1e-7;successful.append(ok);negative.append(bad)
        result['edges'][p+'->'+c]=dict(strict_pct=float(ok.double().mean()*100),mismatched_pct=float(bad.double().mean()*100),
            radial_order_pct=float((r[c]>r[p]).double().mean()*100),violation=float(violation.mean()),
            angle_degrees=float(torch.rad2deg(phi).mean()),aperture_degrees=float(torch.rad2deg(omega).mean()))
    result.update(strict_pct=float(torch.stack(successful).double().mean()*100),mismatched_pct=float(torch.stack(negative).double().mean()*100),
                  all_nine_pct=float(torch.stack(successful).all(0).double().mean()*100))
    u={s:F.normalize(x.double(),dim=-1) for s,x in z.items()}
    caps={s:cap_width(r[s]) if geometry=='hyperbolic' else torch.asin(.2/r[s]) for s in SUBSETS}
    import itertools
    compatible={}
    for child in ('tv','ta','va','tva'):
        parents=[p for p,c in EDGES if c==child]
        gaps=[torch.acos((u[a]*u[b]).sum(-1).clamp(-1+1e-12,1-1e-12))-caps[a]-caps[b] for a,b in itertools.combinations(parents,2)]
        compatible[child]=float((torch.stack(gaps)<=0).all(0).double().mean()*100)
    result['parent_compatibility_necessary_pct']=compatible
    return result


@torch.no_grad()
def retrieval_metrics(features,metric,chunk=256):
    n=len(features['t']);result={};ranks_out={}
    if any(x.shape[0]!=n or not torch.isfinite(x).all() for x in features.values()):raise ValueError('Invalid aligned features')
    for a,b in [('t','v'),('v','t'),('t','a'),('a','t'),('v','a'),('a','v')]:
        x=features[a].double();y=features[b].double();ranks=[]
        if metric=='cosine':x=F.normalize(x,dim=-1);y=F.normalize(y,dim=-1)
        for start in range(0,n,chunk):
            score=x[start:start+chunk]@y.T if metric=='cosine' else -pair_distance(x[start:start+chunk],y,metric)
            index=torch.arange(start,start+len(score),device=score.device)
            positive=score[torch.arange(len(score),device=score.device),index,None]
            rank=(score>positive).sum(-1)+((score==positive)&(torch.arange(n,device=score.device)[None,:]<index[:,None])).sum(-1)+1
            ranks.append(rank.cpu())
        rank=torch.cat(ranks);direction=a+'2'+b;ranks_out[direction]=rank
        result[direction]={f'R@{k}':float((rank<=k).double().mean()*100) for k in (1,5,10)}
        result[direction]['median_rank']=float(rank.double().median())
    return dict(directions=result,macro={f'R@{k}':sum(v[f'R@{k}'] for v in result.values())/6 for k in (1,5,10)}),ranks_out
