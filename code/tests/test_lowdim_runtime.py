"""Failure recovery and resource isolation contracts, with small CPU fixtures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from experiments.subset_hierarchy.lowdim_finetune import commit_epoch,point_best,write_history
from experiments.subset_hierarchy.lowdim_schedule import allocate,options
from experiments.subset_hierarchy.lowdim_models import install_checkpoint_policy
from experiments.subset_hierarchy.lowdim_common import ROOT,resolve_ref


class RuntimeTests(unittest.TestCase):
    def test_commit_failure_keeps_recoverable_previous_best(self):
        (ROOT/'.tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='lowdim_resume_',dir=ROOT/'.tmp') as temp:
            suite=Path(temp);run=suite/'run';run.mkdir();net=torch.nn.Linear(3,2);opt=torch.optim.AdamW(net.parameters())
            scaler=torch.cuda.amp.GradScaler(enabled=False);config={'world_size':2,'seed':50};history=[]
            def commit(epoch,best_file,improved=True):
                summary=dict(epoch=epoch,step=epoch,validation={'macro':{'R@1':float(epoch)}})
                history.append(summary)
                return commit_epoch(suite,run,net,opt,scaler,config,summary,history,[{'rank':0},{'rank':1}],epoch,epoch,best_file,improved)
            with patch('experiments.subset_hierarchy.lowdim_finetune.shutil.disk_usage',return_value=type('Disk',(),{'free':100*2**30})()):
                first=commit(1,None);saved=torch.load(run/'last.pt');before={k:v.clone() for k,v in net.state_dict().items()}
                net(torch.ones(2,3)).square().sum().backward();opt.step();opt.zero_grad()
                from experiments.subset_hierarchy import lowdim_finetune as runner
                real=runner.save_checkpoint
                def fail_last(suite,path,payload):
                    if path.name=='last.pt':raise OSError('simulated interrupted NFS commit')
                    return real(suite,path,payload)
                with patch.object(runner,'save_checkpoint',side_effect=fail_last),self.assertRaises(OSError):commit(2,first)
                self.assertEqual(torch.load(run/'last.pt')['epoch'],1);self.assertTrue((run/first).exists())
                net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);history[:]=saved['history']
                point_best(run,saved['best_file']);write_history(run,saved['history'])
                for k,v in net.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
                second=commit(2,first)
                self.assertEqual((run/'best.pt').resolve(),run/second);self.assertFalse((run/first).exists())
                self.assertEqual(torch.load(run/'last.pt')['rngs'],[{'rank':0},{'rank':1}])
                self.assertEqual(len((run/'epochs.jsonl').read_text().splitlines()),2)
    def test_resource_planner_enforces_headroom_and_slots(self):
        inventory=[dict(index=str(i),free_gib=40) for i in range(4)]
        self.assertEqual(len(allocate(inventory,4,30,[])),4)
        self.assertIsNone(allocate(inventory,4,38,[]))
        active=[dict(devices=['0'],peak_gib=10)]
        self.assertNotIn('0',allocate(inventory,3,10,active))
        self.assertIsNotNone(allocate(inventory,4,10,active,dual=True))
        self.assertIsNone(allocate(inventory,4,10,active*2,dual=True))
        rows=[dict(dataset='msrvtt',family='hyp_cone',world_size=1,samples_per_second=8,dual=True,dual_samples_per_second=18),dict(dataset='msrvtt',family='hyp_cone',world_size=2,samples_per_second=30,dual=False)]
        self.assertEqual(options(rows,'msrvtt','euc_none')[0]['world_size'],1)
    def test_nonreentrant_policy_handles_unused_parameters(self):
        import torch.utils.checkpoint as cp
        original=cp.checkpoint
        try:
            install_checkpoint_policy();module=torch.nn.Linear(3,2);x=torch.ones(2,3)
            cp.checkpoint(module,x).sum().backward()
            self.assertIsNotNone(module.weight.grad)
        finally:cp.checkpoint=original
    def test_annotation_only_bootstrap_is_portable_and_strict(self):
        from types import SimpleNamespace
        from experiments.subset_hierarchy.lowdim_common import write_json,file_hash
        from experiments.subset_hierarchy.lowdim_handoff import bootstrap
        (ROOT/'.tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='lowdim_bootstrap_',dir=ROOT/'.tmp') as temp:
            root=Path(temp);package=root/'package';data=root/'data';authors=root/'authors';suite=data/'suite'
            package.mkdir();data.mkdir()
            protocol=dict(checkpoints={},model_cfg={'default':'/old/server/config'},gram_hps={'model_cfg':{'default':'old'},'unused':True},checkpoint_sha256={m:'pending' for m in ('vast','gram','pmrl')})
            write_json(package/'protocol.json',protocol);write_json(package/'tasks.json',[]);write_json(package/'annotations/manifest.json',{})
            attrs=dict(package=package,assets=data/'assets');attrs['assets'].mkdir()
            for method in ('vast','gram','pmrl'):
                checkpoint=data/(method+'.pt');checkpoint.write_bytes(b'fixture, never loaded as a model');attrs[method+'_checkpoint']=checkpoint
            for method in ('vast','gram'):
                path=authors/method.upper()/'model'/f'{method}.py';path.parent.mkdir(parents=True);path.write_text('# fixture')
            for dataset in ('msrvtt','activitynet'):
                write_json(package/'annotations'/dataset/'candidates.json',{split:[dict(clip_id=split,captions=['fixture'])] for split in ('train','val','test')})
                for kind,extension in [('video','mp4'),('audio','wav')]:
                    path=data/dataset/kind;path.mkdir(parents=True);attrs[dataset+'_'+kind]=path
                    for split in ('train','val','test'):(path/(split+'.'+extension)).touch()
            write_json(package/'SHA256.json',{str(p.relative_to(package)):file_hash(p) for p in package.rglob('*') if p.is_file()})
            roots=dict(workspace=root,data=data,suite=suite,authors=authors,logs=root/'logs');args=SimpleNamespace(**attrs)
            bootstrap(args,roots);bootstrap(args,roots)
            self.assertFalse((suite/'CPU_READY.json').exists());self.assertTrue((suite/'BOOTSTRAP.json').exists())
            saved=json.loads((suite/'protocol.json').read_text());self.assertNotIn('default',saved['model_cfg'])
            (attrs['msrvtt_video']/'train.mp4').unlink()
            with self.assertRaisesRegex(RuntimeError,'required media files missing'):bootstrap(args,roots)

    def test_resource_paths_do_not_escape_roots(self):
        roots={k:ROOT/k for k in ('workspace','suite','data','authors')}
        with self.assertRaises(ValueError):resolve_ref({'root':'data','path':'../../../outside'},roots)

if __name__=='__main__':unittest.main()
