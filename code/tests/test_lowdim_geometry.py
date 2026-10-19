"""Mathematical and initialization contracts for the new low-dimensional suite."""
import math
import unittest
import torch
from utils.subset_hierarchy import SUBSETS,SubsetHeads,pair_distance
from utils.lowdim_geometry import (euclidean_cone_parts,fit_basis,projection,project_native_state,
    initialize_subset_heads,loss,retrieval_metrics)
from experiments.subset_hierarchy.lowdim_data import GlobalBatchSampler


class LowDimTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(50)
    def test_euclidean_cone_cases_and_gradients(self):
        p=torch.tensor([[.5,0.]],dtype=torch.float64)
        for q,expected in [(torch.tensor([[1.,0.]]),True),(torch.tensor([[.25,0.]]),False),(torch.tensor([[.5,1.]]),False),(p,True)]:
            a=p.clone().requires_grad_();b=q.double().clone().requires_grad_();phi,omega=euclidean_cone_parts(a,b)
            e=(phi-omega).relu();self.assertEqual(bool(e.item()==0),expected)
            e.sum().backward();self.assertTrue(torch.isfinite(a.grad).all());self.assertTrue(torch.isfinite(b.grad).all())
        with self.assertRaises(ValueError):euclidean_cone_parts(torch.tensor([[.1,0.]]),p)
        r=.001;k=.0001
        _,ap=euclidean_cone_parts(torch.tensor([[r,0.]],dtype=torch.float64),torch.tensor([[2*r,0.]],dtype=torch.float64),k=k)
        self.assertLess(abs(ap.item()-math.asin(k/math.sinh(r))),1e-7)
    def test_basis_nested_and_identity(self):
        feats={m:torch.randn(64,512) for m in 'tva'};basis,_=fit_basis(feats)
        for d in [8,32,128,512]:
            p=projection(basis,d);torch.testing.assert_close(p@p.T,torch.eye(d),rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(projection(basis,8),projection(basis,32)[:8],rtol=0,atol=0)
        torch.testing.assert_close(projection(basis,512),torch.eye(512),rtol=0,atol=0)
    def test_paired_initialization_all_dimensions(self):
        dims={'t':6,'v':8,'a':5};state={f'contra_head_{m}.linear.weight':torch.randn(512,d) for m,d in dims.items()}
        state['contra_head_va.weight']=torch.randn(512,13);state['contra_head_va.bias']=torch.randn(512)
        basis=torch.eye(512)
        for d in [8,32,128,512]:
            h=initialize_subset_heads(dims,state,basis,d,'hyperbolic');e=initialize_subset_heads(dims,state,basis,d,'euclidean')
            for k,v in h.state_dict().items():torch.testing.assert_close(v,e.state_dict()[k],rtol=0,atol=0)
            x={m:torch.randn(8,n) for m,n in dims.items()};zh=h(x);ze=e(x)
            for s in SUBSETS:
                self.assertEqual(zh[s].shape,(8,d));torch.testing.assert_close(zh[s].norm(dim=-1).asinh(),torch.ones(8,dtype=torch.float64),rtol=1e-6,atol=1e-6)
                torch.testing.assert_close(ze[s].norm(dim=-1),torch.ones(8,dtype=torch.float64),rtol=1e-6,atol=1e-6)
            projected,_=project_native_state(state,basis,d)
            self.assertEqual(projected['contra_head_va.weight'].shape,(d,13))
            if d==512:
                for k,v in state.items():torch.testing.assert_close(v,projected[k],rtol=0,atol=0)
    def test_all_arms_have_finite_head_gradients(self):
        for geometry in ['hyperbolic','euclidean']:
            for weight in [0.,.1]:
                model=SubsetHeads({'t':6,'v':8,'a':5},8,geometry)
                x={m:torch.randn(8,d) for m,d in model.input_dims.items()}
                value,metrics=loss(model(x),geometry,weight,distributed=False);value.backward()
                self.assertTrue(torch.isfinite(value));self.assertEqual(int(metrics['candidate_count']),8)
                for p in model.parameters():self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
    def test_stable_retrieval_ranks_equal_sort(self):
        for metric in ['cosine','hyperbolic','euclidean']:
            features={m:torch.randn(13,8,dtype=torch.float64) for m in 'tva'}
            if metric=='hyperbolic':features={m:x/x.norm(dim=-1,keepdim=True)*math.sinh(1) for m,x in features.items()}
            actual,ranks=retrieval_metrics(features,metric,chunk=3)
            for d in ranks:
                a,b=features[d[0]],features[d[-1]]
                scores=torch.nn.functional.normalize(a,dim=-1)@torch.nn.functional.normalize(b,dim=-1).T if metric=='cosine' else -pair_distance(a,b,metric)
                expected=(scores.argsort(dim=1,descending=True,stable=True)==torch.arange(13)[:,None]).nonzero()[:,1]+1
                torch.testing.assert_close(ranks[d],expected)
        features={m:torch.ones(4,8) for m in 'tva'}
        metrics,_=retrieval_metrics(features,'cosine');self.assertEqual(metrics['macro']['R@1'],25.)
    def test_global_batches_do_not_depend_on_world_size(self):
        full=list(GlobalBatchSampler(131,batch=64,seed=50,epoch=3))
        for world in [2,4,8]:
            pieces=[list(GlobalBatchSampler(131,rank=r,world=world,batch=64,seed=50,epoch=3)) for r in range(world)]
            self.assertEqual(full,[sum([pieces[r][b] for r in range(world)],[]) for b in range(len(full))])
        evals=[list(GlobalBatchSampler(131,rank=r,world=4,training=False)) for r in range(4)]
        ids=[i for rank in evals for batch in rank for i in batch];self.assertEqual(sorted(ids),list(range(131)))

if __name__=='__main__':unittest.main()
