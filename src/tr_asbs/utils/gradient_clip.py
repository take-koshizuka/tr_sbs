"""Recover finite dense gradients when their float32 norm overflows."""
import torch


def clip_gradients_with_overflow_fallback(fabric, model, optimizer, max_norm):
    """Keep Fabric clipping normally; retry only its finite-norm overflow.

    Fabric has already unscaled gradients before its norm check. This fallback
    supports one process and dense gradients; genuine nonfinite gradients and
    distributed/sharded clipping retain the original rejection behavior.
    """
    try:
        return fabric.clip_gradients(model, optimizer, max_norm=max_norm,
                                     error_if_nonfinite=True)
    except RuntimeError as error:
        if 'non-finite' not in str(error) or fabric.world_size != 1:
            raise
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        if (not gradients or any(g.is_sparse or not torch.isfinite(g).all()
                                 for g in gradients)):
            raise
        norm=torch.sqrt(sum(g.double().square().sum() for g in gradients))
        if not torch.isfinite(norm):
            raise
        scale=(max_norm/(norm+1e-6)).clamp(max=1.)
        with torch.no_grad():
            # Multiply before casting so a very small scale cannot underflow.
            for gradient in gradients:
                gradient.copy_((gradient.double()*scale).to(gradient.dtype))
        return norm
