"""Intrinsic Lorentz tuple geometry, using spatial coordinates and FP64.

All inputs represent the upper sheet; the time coordinate is reconstructed.
The distance is exact outside a 1e-12 neighbourhood of coincidence, where a
continuous finite-slope square root gives zero distance and finite gradients.
"""
import math

import torch


EPS = 1e-12
MAX_SCALED_RADIUS = math.asinh(2**15)


def time_component(x, curvature):
    return (curvature.reciprocal() + x.square().sum(-1, keepdim=True)).sqrt()


def expmap0(tangent, curvature):
    tangent = tangent.double()
    curvature = curvature.double()
    radius = curvature.sqrt() * torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    shrink = (MAX_SCALED_RADIUS / radius.clamp_min(EPS)).clamp(max=1.0)
    tangent = tangent * shrink
    radius = radius * shrink
    factor = torch.where(radius < 1e-6, 1 + radius.square() / 6,
                         torch.sinh(radius) / radius.clamp_min(EPS))
    return factor * tangent


def pair_inner(x, y, curvature):
    """-K <x_i,y_j>_L, without constructing an N*M*D tensor."""
    x, y, curvature = x.double(), y.double(), curvature.double()
    tx, ty = time_component(x, curvature), time_component(y, curvature)
    return (curvature * (tx @ ty.transpose(-1, -2) - x @ y.transpose(-1, -2))).clamp_min(1)


def matched_inner(x, y, curvature):
    x, y, curvature = x.double(), y.double(), curvature.double()
    return (curvature * (time_component(x, curvature).squeeze(-1)
                        * time_component(y, curvature).squeeze(-1)
                        - (x * y).sum(-1))).clamp_min(1)


def tuple_gram(points, curvature):
    matrix = pair_inner(points, points, curvature)
    eye = torch.eye(points.shape[-2], dtype=matrix.dtype, device=matrix.device)
    return matrix * (1 - eye) + eye


def candidate_gram(query, candidates, curvature):
    """Gram matrices for each query and each *paired* candidate tuple.

    query: [N,D]; candidates: sequence of [M,D]; return [N,M,k,k].
    """
    if not candidates or any(x.shape != candidates[0].shape for x in candidates):
        raise ValueError("candidate modalities must be nonempty and share shape")
    cross = [pair_inner(query, x, curvature) for x in candidates]
    one = torch.ones_like(cross[0])
    rows = [torch.stack([one, *cross], dim=-1)]
    for i, x in enumerate(candidates):
        row = [cross[i]]
        for j, y in enumerate(candidates):
            value = one if i == j else matched_inner(x, y, curvature).unsqueeze(0).expand_as(one)
            row.append(value)
        rows.append(torch.stack(row, dim=-1))
    return torch.stack(rows, dim=-2)


def _sqrt_near_zero(delta):
    delta = delta.clamp_min(0)
    return torch.where(delta >= EPS, delta.clamp_min(EPS).sqrt(), delta / math.sqrt(EPS))


def distance_from_eigenvalue(value, k, curvature):
    # acosh(sqrt(value/k)) == asinh(sqrt(value/k - 1)); no cosh of a distance.
    return 2 * torch.asinh(_sqrt_near_zero((value - k) / k)) / curvature.sqrt()


def pair_distance(x, y, curvature):
    inner = pair_inner(x, y, curvature)
    return 2 * torch.asinh(_sqrt_near_zero((inner - 1) / 2)) / curvature.double().sqrt()


def gram_distance(matrix, curvature, score="spectral"):
    k = matrix.shape[-1]
    if score == "spectral":
        value = torch.linalg.eigvalsh(matrix)[..., -1]
    elif score == "uniform":
        value = matrix.sum(dim=(-2, -1)) / k
    else:
        raise ValueError(f"Unknown Lorentz score: {score}")
    return distance_from_eigenvalue(value, k, curvature)


class _DominantEigenpair(torch.autograd.Function):
    """Differentiate only the simple Perron eigenpair, not lower eigenspaces."""

    @staticmethod
    def forward(ctx, matrix):
        values, vectors = torch.linalg.eigh(matrix)
        value, vector = values[..., -1], vectors[..., -1]
        vector = vector * torch.where(vector.sum(-1, keepdim=True) < 0, -1., 1.)
        ctx.save_for_backward(matrix, value, vector)
        return value, vector

    @staticmethod
    def backward(ctx, grad_value, grad_vector):
        matrix, value, vector = ctx.saved_tensors
        outer = vector.unsqueeze(-1) * vector.unsqueeze(-2)
        gradient = torch.zeros_like(matrix)
        if grad_value is not None:
            gradient = grad_value[..., None, None] * outer
        if grad_vector is not None:
            # On u-perp the inverse is (lambda I-M)^-1.  Adding lambda uu^T
            # makes the system invertible and well-conditioned on all of R^k.
            rhs = grad_vector - vector * (vector * grad_vector).sum(-1, keepdim=True)
            eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
            system = value[..., None, None] * (eye + outer) - matrix
            solution = torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)
            term = solution.unsqueeze(-1) * vector.unsqueeze(-2)
            gradient = gradient + (term + term.transpose(-1, -2)) / 2
        return gradient


def dominant_eigenpair(matrix):
    return _DominantEigenpair.apply((matrix + matrix.transpose(-1, -2)) / 2)


def consensus(points, curvature):
    if points.shape[-2] == 1:
        return points[..., 0, :]
    value, vector = dominant_eigenpair(tuple_gram(points, curvature))
    return (vector.unsqueeze(-1) * points).sum(-2) / value.sqrt().unsqueeze(-1)


def origin_radius(points, curvature):
    return torch.asinh(curvature.sqrt() * points.norm(dim=-1)) / curvature.sqrt()
