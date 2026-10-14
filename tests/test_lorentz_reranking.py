"""The existing FP32 ITM must rerank either FP32 or FP64 candidate scores."""
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from evaluation import evaluation_mm as evaluation


class RerankingTests(unittest.TestCase):
    def test_candidate_dtype_preserves_native_itm_values_in_both_directions(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
            def compute_slice_scores(self, condition, ids, attention):
                return torch.sigmoid(ids[:, 0].float() * .1 + condition[:, 0, 0].float() * .03)
        model = Model()
        inputs = torch.arange(4)[:, None]
        condition = torch.arange(4).float()[:, None, None]
        base = torch.tensor([[9, 8, 1, 2], [4, 7, 8, 6], [5, 1, 9, 3], [6, 2, 4, 8.]])
        bar = SimpleNamespace(update=lambda *a: None, close=lambda: None)
        with mock.patch.object(evaluation.dist, 'get_rank', return_value=0), \
             mock.patch.object(evaluation.dist, 'get_world_size', return_value=1), \
             mock.patch.object(evaluation, 'all_gather_list', side_effect=lambda x: [x]), \
             mock.patch.object(evaluation, 'ddp_allgather', side_effect=lambda x: x), \
             mock.patch.object(evaluation, 'tqdm', return_value=bar):
            for dtype in (torch.float32, torch.float64):
                scores = base.to(dtype)
                for direction in ('forward', 'backward'):
                    result = evaluation.refine_score_matrix(condition, inputs, torch.ones_like(inputs),
                                                             scores, model, 2, direction)
                    mask = torch.zeros_like(scores, dtype=torch.bool)
                    axis = 1 if direction == 'forward' else 0
                    mask.scatter_(axis, scores.topk(2, dim=axis).indices, True)
                    expected = torch.zeros_like(scores)
                    for i in range(4):
                        for j in range(4):
                            if mask[i, j]:
                                expected[i, j] = model.compute_slice_scores(condition[j:j+1], inputs[i:i+1], None)[0]
                    self.assertEqual(result.dtype, dtype)
                    torch.testing.assert_close(result, expected)


if __name__ == '__main__':
    unittest.main()
