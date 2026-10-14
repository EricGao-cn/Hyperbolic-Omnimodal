"""CPU geometry/objective tests; no encoder construction or model downloads."""
import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lorentz_omnimodal import LorentzOmnimodal
from utils import lorentz_spectral as g


def small_model(**updates):
    config = SimpleNamespace(contra_dim=8, **updates)
    model = LorentzOmnimodal.__new__(LorentzOmnimodal)
    nn.Module.__init__(model)
    model.config = config
    model._init_geometry(config)
    return model


class LorentzTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(50)
        self.c = torch.tensor(1., dtype=torch.double)

    def points(self, *shape):
        return g.expmap0(torch.randn(*shape, dtype=torch.double) * .25, self.c)

    def test_pair_reduction_and_spectral_properties(self):
        for k in (2, 3, 4):
            points = self.points(7, k, 8)
            matrix = g.tuple_gram(points, self.c)
            values = torch.linalg.eigvalsh(matrix)
            self.assertTrue((values[:, -1] >= k - 1e-10).all())
            self.assertTrue((values[:, -2] <= 1e-10).all())
            self.assertTrue(torch.allclose(matrix.diagonal(dim1=-2, dim2=-1), torch.ones(7, k, dtype=torch.double)))
            if k == 2:
                expected = g.pair_distance(points[:, 0], points[:, 1], self.c).diag()
                self.assertTrue(torch.allclose(g.gram_distance(matrix, self.c), expected, atol=1e-10))

    def test_permutation_and_isometry(self):
        x = self.points(3, 4, 8)
        original = g.gram_distance(g.tuple_gram(x, self.c), self.c)
        permuted = g.gram_distance(g.tuple_gram(x[:, [3, 1, 0, 2]], self.c), self.c)
        self.assertTrue(torch.allclose(original, permuted, atol=1e-12))
        # A Lorentz boost, not just an origin-preserving rotation.
        boost = .6
        y = x.clone()
        y[..., 0] = math.sinh(boost) * g.time_component(x, self.c).squeeze(-1) + math.cosh(boost) * x[..., 0]
        self.assertTrue(torch.allclose(g.tuple_gram(x, self.c), g.tuple_gram(y, self.c), atol=1e-12))

    def test_consensus_manifold_and_identity(self):
        x = self.points(6, 3, 8)
        matrix = g.tuple_gram(x, self.c)
        value, u = g.dominant_eigenpair(matrix)
        self.assertTrue((u > 0).all())
        mu = g.consensus(x, self.c)
        expected_time = (u * g.time_component(x, self.c).squeeze(-1)).sum(-1) / value.sqrt()
        self.assertTrue(torch.allclose(g.time_component(mu, self.c).squeeze(-1), expected_time, atol=1e-12))
        self.assertTrue(torch.equal(g.consensus(x[:, :1], self.c), x[:, 0]))
        measured = g.matched_inner(mu[:, None], x, self.c)
        self.assertTrue(torch.allclose(measured, value.sqrt()[:, None] * u, atol=1e-12))

    def test_collinear_points_are_not_aligned(self):
        x = g.expmap0(torch.tensor([[.2, 0.], [1., 0.], [2., 0.]], dtype=torch.double), self.c)
        matrix = g.tuple_gram(x, self.c)
        self.assertLess(abs(float(torch.linalg.det(matrix))), 1e-10)
        self.assertGreater(float(g.gram_distance(matrix, self.c)), .1)

    def test_candidate_block_matches_direct_tuples(self):
        q, v, a = [self.points(5, 8) for _ in range(3)]
        actual = g.candidate_gram(q, [v, a], self.c)
        expected = torch.stack([torch.stack([g.tuple_gram(torch.stack([t, x, y]), self.c)
                                           for x, y in zip(v, a)]) for t in q])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12))

    def test_dominant_eigenpair_gradcheck_and_repeated_lower_spectrum(self):
        for k in (3, 4):
            raw = (torch.ones(k, k, dtype=torch.double) * 16 - torch.eye(k, dtype=torch.double) * 15).requires_grad_()
            self.assertTrue(torch.autograd.gradcheck(g.dominant_eigenpair, (raw,), atol=1e-5))
            value, vector = g.dominant_eigenpair(raw)
            gradient = torch.autograd.grad(value + (vector * torch.arange(k)).sum(), raw)[0]
            self.assertTrue(torch.isfinite(gradient).all())

    def test_distance_and_consensus_gradients(self):
        z = (torch.randn(2, 3, 4, dtype=torch.double) * .2).requires_grad_()
        curvature = torch.tensor(.7, dtype=torch.double, requires_grad=True)
        def objective(z, c):
            x = g.expmap0(z, c)
            return g.gram_distance(g.tuple_gram(x, c), c).sum() + g.consensus(x, c).square().sum()
        self.assertTrue(torch.autograd.gradcheck(objective, (z, curvature), atol=1e-5))

    def test_coincident_and_large_radius_finite(self):
        for c_value in (.1, 1., 10.):
            c = torch.tensor(c_value, dtype=torch.double, requires_grad=True)
            for radius in (0., .2, 20.):
                z = torch.zeros(2, 3, 4, dtype=torch.double)
                z[..., 0] = radius
                z.requires_grad_()
                x = g.expmap0(z, c)
                loss = g.gram_distance(g.tuple_gram(x, c), c).sum() + g.consensus(x, c).square().mean()
                gradients = torch.autograd.grad(loss, (z, c))
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(v).all() for v in gradients))

    def test_uniform_is_rayleigh_control(self):
        matrix = g.tuple_gram(self.points(3, 4, 8), self.c)
        expected = g.distance_from_eigenvalue(matrix.sum((-2, -1)) / 4, 4, self.c)
        self.assertTrue(torch.equal(g.gram_distance(matrix, self.c, "uniform"), expected))

    def test_objectives_and_all_modality_gradients(self):
        for score in ("spectral", "uniform", "allpair"):
            model = small_model(hyperbolic_score=score)
            inputs = [torch.randn(5, 8, requires_grad=True) for _ in range(3)]
            points = [model._lift(x, m) for x, m in zip(inputs, "tva")]
            loss = model.spectral_loss(*points)
            if score == "allpair":
                expected = sum(model._geodesic_contrastive(points[a], points[b]) for a, b in ((0, 1), (0, 2), (1, 2))) / 3
            else:
                matrix = g.candidate_gram(points[0], points[1:], model.hyperbolic_curvature)
                logits = -g.gram_distance(matrix, model.hyperbolic_curvature, score) / model.hyperbolic_spectral_temperature
                expected = (F.cross_entropy(logits, torch.arange(5)) + F.cross_entropy(logits.T, torch.arange(5))) / 2
            self.assertTrue(torch.allclose(loss, expected, atol=1e-10))
            loss = loss + .1 * model.consensus_loss(*points)
            gradients = torch.autograd.grad(loss, inputs)
            self.assertTrue(all(torch.isfinite(x).all() and x.norm() > 0 for x in gradients))

    def test_native_chunking_and_invalid_configuration(self):
        model = small_model()
        q, v, a = [self.points(5, 8) for _ in range(3)]
        self.assertTrue(torch.allclose(model.retrieval_tuple_similarity(q, dict(v=v, a=a), 2),
                                       model.retrieval_tuple_similarity(q, dict(v=v, a=a), 5)))
        with self.assertRaises(ValueError):
            small_model(hyperbolic_entailment_weight=-.1)
        self.assertEqual(small_model(hyperbolic_entailment_weight=.1).hyperbolic_entailment_weight, .1)
        with self.assertRaises(ValueError):
            small_model(hyperbolic_temperature=0)
        self.assertEqual(len(model._subset_pairs), 12)

    def test_old_mapping_has_finite_trainable_heads(self):
        model = small_model(hyperbolic_mapping='direction_radius')
        points = [model._lift(torch.randn(4, 8), m) for m in 'tva']
        model.spectral_loss(*points).backward()
        for heads in (model.hyperbolic_direction_heads, model.hyperbolic_radius_heads):
            self.assertTrue(all(torch.isfinite(p.grad).all() and p.grad.norm() > 0
                                for p in heads.parameters()))


if __name__ == "__main__":
    unittest.main()
