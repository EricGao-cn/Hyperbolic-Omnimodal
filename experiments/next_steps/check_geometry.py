"""CPU checks for the claims the method rests on and for the new loss paths.

Run from the repository root before launching the suite:
    python experiments/next_steps/check_geometry.py
Exits non-zero on the first failed check. No GPU, data or weights needed.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from easydict import EasyDict
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import lorentz_spectral as geometry  # noqa: E402
from model.lorentz_omnimodal import LorentzOmnimodal  # noqa: E402
from model.hyperbolic_spectral_pmrl import HyperbolicSpectralPMRL  # noqa: E402
from model.pmrl import PMRL  # noqa: E402

torch.manual_seed(0)


def random_points(batch, k, dim, curvature, scale=1.):
    tangent = torch.randn(batch, k, dim) * scale / math.sqrt(dim) / curvature.sqrt()
    return geometry.expmap0(tangent, curvature)


def check(name, condition, detail=''):
    print(('ok    ' if condition else 'FAIL  ') + name + (f'  [{detail}]' if detail else ''))
    if not condition:
        raise SystemExit(1)


def check_spectrum():
    worst_gap, worst_identity, worst_manifold, worst_radius = math.inf, 0., 0., -math.inf
    for c_value in (.1, .3, 1., 3., 10.):
        c = torch.tensor(c_value)
        for k in (2, 3, 4, 5):
            x = random_points(512, k, 16, c, scale=2.)
            matrix = geometry.tuple_gram(x, c)
            values = torch.linalg.eigvalsh(matrix)
            check(f'P1 one positive eigenvalue, lambda>=k (c={c_value}, k={k})',
                  bool((values[..., :-1] <= 1e-9 * values[..., -1:]).all() and (values[..., -1] >= k - 1e-9).all()))
            d = geometry.perron_diagnostics(x, c)
            worst_gap = min(worst_gap, d['perron_gap_over_k_min'])
            worst_identity = max(worst_identity, d['perron_identity_rel_error_max'])
            worst_manifold = max(worst_manifold, d['perron_manifold_error_max'])
            proxy = geometry.consensus(x, c)
            worst_radius = max(worst_radius, (geometry.origin_radius(proxy, c)
                                              - geometry.origin_radius(x, c).max(-1).values).max().item())
            check(f'P3 Perron vector strictly positive (c={c_value}, k={k})', d['perron_min_vector_entry'] > 0)
    check('P2 spectral gap >= k', worst_gap >= 1 - 1e-9, f'min gap/k {worst_gap:.4f}')
    check('P3 proxy on the hyperboloid', worst_manifold < 1e-8, f'{worst_manifold:.1e}')
    check('P3 r(P) <= max_m r(X_m)', worst_radius <= 1e-9, f'{worst_radius:.1e}')
    check('P4 lambda = sum cosh^2 d(P, X_m)', worst_identity < 1e-10, f'{worst_identity:.1e}')
    aligned = random_points(4, 1, 16, torch.tensor(1.)).expand(4, 3, 16)
    check('P1 lambda = k iff aligned', torch.allclose(torch.linalg.eigvalsh(geometry.tuple_gram(aligned, torch.tensor(1.)))[..., -1],
                                                       torch.full((4,), 3., dtype=torch.float64)))


def check_transforms():
    for c_value in (.3, 1., 3.):
        c = torch.tensor(c_value)
        x = random_points(256, 2, 16, c)
        d = geometry.pair_distance(x[:, 0], x[:, 1], c).diagonal()
        m = geometry.tuple_gram(x, c)
        check(f'P5 -D_M equals -geodesic distance at k=2 (c={c_value})',
              torch.allclose(geometry.gram_similarity(m, c, transform='distance'), -d, atol=1e-7))
        check(f'P5 k/lambda equals sech^2(sqrt(c) d/2) at k=2 (c={c_value})',
              torch.allclose(geometry.gram_similarity(m, c, transform='inverse'),
                             torch.cosh(c.sqrt() * d / 2) ** -2, atol=1e-9))
    x = random_points(64, 3, 16, torch.tensor(1.))
    m = geometry.tuple_gram(x, torch.tensor(1.))
    sims = [geometry.gram_similarity(m, torch.tensor(1.), transform=t) for t in geometry.SCORE_TRANSFORMS]
    order = [s.argsort() for s in sims]
    check('all score transforms rank tuples identically', all(torch.equal(order[0], o) for o in order[1:]))
    check('inverse transform lies in (0, 1]', bool(((sims[1] > 0) & (sims[1] <= 1 + 1e-12)).all()))


def check_eigenpair_gradient():
    c = torch.tensor(1.)
    # Coincident V and A make the lower spectrum degenerate: generic eigh backward is NaN there.
    base = random_points(1, 2, 8, c)
    x = torch.cat([base, base[:, 1:]], dim=1).squeeze(0).requires_grad_(True)
    value, vector = geometry.dominant_eigenpair(geometry.tuple_gram(x, c))
    (value + vector @ torch.arange(1., 4., dtype=torch.float64)).backward()
    check('Perron backward finite with a degenerate lower spectrum', bool(torch.isfinite(x.grad).all()))
    y = random_points(1, 3, 6, c).squeeze(0).requires_grad_(True)
    check('Perron eigenpair gradcheck', torch.autograd.gradcheck(
        lambda z: geometry.consensus(z, c), (y,), eps=1e-6, atol=1e-5))


def tiny_model(**overrides):
    config = EasyDict(contra_dim=8, hyperbolic_mapping='direction_radius', hyperbolic_temperature=.07,
                      hyperbolic_curvature_init=1., lambda_itm=.1)
    config.update(overrides)
    model = LorentzOmnimodal.__new__(LorentzOmnimodal)
    nn.Module.__init__(model)
    model._init_geometry(config)
    model.train()
    return model


def check_losses():
    feats = {m: torch.randn(16, 8, dtype=torch.float32) for m in 'tva'}
    for transform in geometry.SCORE_TRANSFORMS:
        for contrast in ('text_vs_va', 'symmetric'):
            model = tiny_model(hyperbolic_score_transform=transform, hyperbolic_contrast=contrast)
            points = [model._lift(feats[m], m) for m in 'tva']
            loss = model.spectral_loss(*points)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            check(f'tuple loss finite with gradients ({transform}, {contrast})',
                  bool(torch.isfinite(loss)) and len(grads) > 0 and all(torch.isfinite(g).all() for g in grads))
    model = tiny_model()
    points = [model._lift(feats[m], m) for m in 'tva']
    reference = model.spectral_loss(*points).item()
    check('default settings unchanged (-D_M, text_vs_va) matches 0929 formula', math.isclose(reference, legacy_loss(model, points), rel_tol=1e-12))
    check('training diagnostics include the Perron checks',
          {'perron_gap_over_k_min', 'perron_identity_rel_error_max', 'positive_tuple_distance'} <= set(model.training_diagnostics()))
    for parent, extra in (('perron', {}), ('modality_edges', dict(hyperbolic_entailment_edges=[['t', 'v'], ['t', 'a']])),
                          ('tangent_mean', {}), ('svd_learned_radius', {})):
        model = tiny_model(hyperbolic_entailment_weight=.1, hyperbolic_entailment_parent=parent, **extra)
        points = [model._lift(feats[m], m) for m in 'tva']
        loss = model.entailment_loss(*points)
        loss.backward()
        sat = [k for k in model.training_diagnostics() if k.startswith('cone_satisfied_')]
        check(f'entailment loss finite, cone satisfaction logged ({parent})', bool(torch.isfinite(loss)) and len(sat) >= 2)
    model = tiny_model(hyperbolic_entailment_weight=.1, hyperbolic_entailment_parent='perron')
    x = [model._lift(feats[m], m) for m in 'tva']
    parent = model.perron_parent(*x)
    c = model.hyperbolic_curvature
    floor = math.asinh(.2 * 1.001) / c.sqrt().item()
    check('Perron parent respects the cone-domain floor',
          bool((geometry.origin_radius(parent, c) >= floor - 1e-9).all()))


def legacy_loss(model, points):
    import torch.nn.functional as F
    text, vision, audio = points
    c = model.hyperbolic_curvature
    targets = torch.arange(text.shape[0])
    forward = -geometry.gram_distance(geometry.candidate_gram(text, [vision, audio], c), c)
    backward = -geometry.gram_distance(geometry.candidate_gram(text, [vision, audio], c), c).T
    tau = model.hyperbolic_spectral_temperature
    return (.5 * (F.cross_entropy(forward / tau, targets) + F.cross_entropy(backward / tau, targets))).item()


def check_matched_kernel_lift():
    config = EasyDict(contra_dim=8, hyperbolic_geometry_fp64=True,
                      hyperbolic_entailment_weight=0., hyperbolic_parent_contraction=1.)
    with mock.patch.object(PMRL, '__init__', lambda self, cfg: nn.Module.__init__(self)):
        kernel = HyperbolicSpectralPMRL(config)
    lorentz = tiny_model()
    features = {m: torch.randn(8, 8) for m in 'tva'}
    for m in 'tva':
        with torch.no_grad():
            kernel.hyperbolic_radius_heads[m].weight.normal_(std=.1)
            lorentz.hyperbolic_radius_heads[m].load_state_dict(kernel.hyperbolic_radius_heads[m].state_dict())
        left, right = kernel._lift(features[m], m), lorentz._lift(features[m], m)
        check(f'Gate A identical FP64 lift ({m})', left.dtype == torch.float64 and torch.equal(left, right))
    points = [kernel._lift(features[m], m) for m in 'tva']
    torch.testing.assert_close(kernel._pairwise_distance(points[0], points[1]),
                               lorentz._pairwise_distance(points[0], points[1]), atol=0, rtol=0)
    loss = kernel.spectral_loss(*points)
    loss.backward()
    check('FP64 kernel loss and gradients finite', loss.dtype == torch.float64
          and torch.isfinite(loss).item() and all(torch.isfinite(p.grad).all() for p in kernel.parameters() if p.grad is not None))
    q, v, a = [lorentz._lift(torch.randn(n, 8), m) for n, m in ((5, 't'), (513, 'v'), (513, 'a'))]
    with torch.no_grad():
        expected = geometry.gram_similarity(geometry.candidate_gram(q, [v, a], lorentz.hyperbolic_curvature),
                                             lorentz.hyperbolic_curvature)
        actual = lorentz.retrieval_tuple_similarity(q, {'v': v, 'a': a}, chunk_size=2)
    check('Native tuple candidate chunking preserves scores', torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))


if __name__ == '__main__':
    check_spectrum()
    check_transforms()
    check_eigenpair_gradient()
    check_losses()
    check_matched_kernel_lift()
    print('all checks passed')
