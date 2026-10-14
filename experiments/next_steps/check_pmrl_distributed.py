"""Check matched PMRL's global negatives and remote gradients on two CPU ranks."""
from pathlib import Path
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from model.pmrl import PMRL


def worker(rank, store):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + store, rank=rank, world_size=2)
    try:
        torch.manual_seed(52)
        full = torch.randn(6, 8, 3, dtype=torch.float64)
        local = full[rank * 3:(rank + 1) * 3].clone().requires_grad_()
        proxy = local[:, :, 0]
        assert not proxy.is_contiguous()
        gathered, labels = PMRL._distributed_pairwise_batch(proxy)
        torch.testing.assert_close(labels, torch.arange(3) + rank * 3)
        loss = F.cross_entropy(proxy @ gathered.T / .1, labels)
        actual_gradient, = torch.autograd.grad(loss, local)
        mean_loss = loss.detach().clone()
        dist.all_reduce(mean_loss)
        mean_loss /= 2
        reference = full.clone().requires_grad_()
        p = reference[:, :, 0]
        expected = F.cross_entropy(p @ p.T / .1, torch.arange(6))
        expected_gradient, = torch.autograd.grad(expected, reference)
        torch.testing.assert_close(mean_loss, expected, atol=1e-10, rtol=1e-8)
        torch.testing.assert_close(actual_gradient / 2, expected_gradient[rank * 3:(rank + 1) * 3],
                                   atol=1e-9, rtol=1e-7)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    (REPO / '.tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=REPO / '.tmp') as temporary:
        mp.spawn(worker, args=(str(Path(temporary) / 'gloo_store'),), nprocs=2, join=True)
    print('PMRL global candidate labels, loss, and nonlocal gradients match the single-process reference')
