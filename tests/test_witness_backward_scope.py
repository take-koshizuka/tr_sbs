import torch

def test_finite_forward_nonfinite_backward_and_scaling_is_diagnostic():
    x=torch.tensor(1e10,requires_grad=True);y=torch.tensor(1e30)*torch.exp(x*x-(x*x).detach());assert torch.isfinite(y);y.backward();assert not torch.isfinite(x.grad)

def test_reduction64_does_not_change_native_inputs():
    x=torch.tensor([1e20,0.]);assert not torch.isfinite(x.var());assert torch.isfinite(x.double().var());assert x.dtype==torch.float32
