"""Binary loss scaling with the original clipped-gradient semantics.

Production use is an explicit LV-controller option. Normalization is undone
before clipping; saved-batch diagnostics use the same arithmetic.
"""
import math

import torch


def power_of_two_scale(loss):
    """Normalize a finite scalar loss toward magnitude one; no tuned constant."""
    value = float(loss.detach())
    if not math.isfinite(value):
        raise ValueError('Nonfinite loss is not repaired by backward scaling')
    exponent = max(0, math.ceil(math.log2(abs(value)))) if abs(value) > 1 else 0
    return math.ldexp(1., -exponent)


@torch.no_grad()
def unscale_and_clip(parameters, scale, max_norm):
    """Compute clip(g_scaled / scale) in FP64, then cast the clipped vector."""
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('Invalid backward scale')
    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError('Invalid norm bound')
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients or any(g.is_sparse or not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError('Scaled gradient is non-finite or unsupported')
    norm = torch.sqrt(sum(g.double().square().sum() for g in gradients)) / scale
    if not torch.isfinite(norm):
        raise RuntimeError('Unscaled gradient norm is non-finite')
    factor = (max_norm / (norm + 1e-6)).clamp(max=1.)
    for gradient in gradients:
        gradient.copy_((gradient.double() / scale * factor).to(gradient.dtype))
    return norm
