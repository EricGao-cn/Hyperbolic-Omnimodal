"""Unit leading right singular vector; only its leading eigengap enters backward."""
import torch


class LeadingSVD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix):
        _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        gap = singular[..., 0].square() - singular[..., 1].square()
        if bool((gap <= 1e-5).any()):
            raise FloatingPointError('SVD proxy leading squared-singular-value gap <= 1e-5')
        ctx.save_for_backward(matrix, singular, vh)
        return vh[..., 0, :]

    @staticmethod
    def backward(ctx, gradient):
        matrix, singular, vh = ctx.saved_tensors
        q = vh[..., 0, :]
        g = gradient - (gradient * q).sum(-1, keepdim=True) * q
        coefficients = torch.einsum('...kd,...d->...k', vh[..., 1:, :], g)
        projected = torch.einsum('...k,...kd->...d', coefficients, vh[..., 1:, :])
        gaps = singular[..., :1].square() - singular[..., 1:].square()
        h = torch.einsum('...k,...kd->...d', coefficients / gaps, vh[..., 1:, :])
        h = h + (g - projected) / singular[..., :1].square()
        zh = torch.einsum('...kd,...d->...k', matrix, h)
        zq = torch.einsum('...kd,...d->...k', matrix, q)
        return zh.unsqueeze(-1) * q.unsqueeze(-2) + zq.unsqueeze(-1) * h.unsqueeze(-2)


def svd_proxy(directions):
    """Input [..., modalities, dimensions]; orient toward the equal direction mean."""
    q = LeadingSVD.apply(directions)
    orientation = (q * directions.mean(-2)).sum(-1, keepdim=True)
    if bool((orientation.abs() <= 1e-6).any()):
        raise FloatingPointError('SVD proxy orientation is ambiguous')
    return q * torch.where(orientation >= 0, 1.0, -1.0)
