"""Small finite-range regressions; no molecular simulations or large models."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from tr_asbs.experiment.jax_inner import AdamSpec,adam_init,adam_update,straight_through_clip
from tr_asbs.model.mlp import clip
from tr_asbs.utils.gradient_clip import clip_gradients_with_overflow_fallback

jax.config.update('jax_enable_x64',True)


def fabric():
    return SimpleNamespace(world_size=1,clip_gradients=lambda model,optimizer,**kwargs:
        torch.nn.utils.clip_grad_norm_(model.parameters(),kwargs['max_norm'],error_if_nonfinite=True))


def test_st_clip_keeps_exact_bound_and_identity_derivative_for_large_values():
    value=torch.tensor([1e8,1e12,1e20,-1e20,0.],requires_grad=True)
    result=clip(value,100.);result.sum().backward()
    torch.testing.assert_close(result,torch.tensor([100.,100.,100.,-100.,0.]),rtol=0,atol=0)
    torch.testing.assert_close(value.grad,torch.ones_like(value),rtol=0,atol=0)
    update=jax.jit(lambda x:straight_through_clip(x,100.))
    x=jnp.asarray(value.detach().numpy())
    np.testing.assert_array_equal(update(x),result.detach().numpy())
    np.testing.assert_array_equal(jax.grad(lambda x:update(x).sum())(x),np.ones(5,dtype=np.float32))
    torch.testing.assert_close(torch.func.jacrev(lambda x:clip(x,100.))(value.detach()),torch.eye(5))
    _,tangent=torch.func.jvp(lambda x:clip(x,100.),(value.detach(),),(torch.ones_like(value),))
    torch.testing.assert_close(tangent,torch.ones_like(value))


@pytest.mark.parametrize('magnitude',[3e19,3e38])
def test_finite_overflow_clipping_and_adam_match_float64_reference(magnitude):
    model=torch.nn.Linear(3,1,bias=False)
    model.weight.data.copy_(torch.tensor([[.4,-.2,.7]]))
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    gradient=torch.tensor([[magnitude,-magnitude,magnitude]],dtype=torch.float32)
    model.weight.grad=gradient.clone()
    norm=clip_gradients_with_overflow_fallback(fabric(),model,optimizer,100.)
    expected=gradient.double()*(100./(gradient.double().norm()+1e-6))
    assert torch.isfinite(norm)
    torch.testing.assert_close(model.weight.grad,expected.float(),rtol=1e-6,atol=1e-6)
    params={'p':jnp.asarray(model.weight.detach().numpy())};state=adam_init(params)
    (updated,new_state),jax_norm,valid=adam_update(params,state,{'p':jnp.asarray(gradient.numpy())},1.,AdamSpec())
    optimizer.step()
    assert bool(valid) and int(new_state.count)==1
    np.testing.assert_allclose(jax_norm,norm.numpy(),rtol=1e-12)
    np.testing.assert_allclose(updated['p'],model.weight.detach().numpy(),rtol=1e-6,atol=1e-7)
    assert updated['p'].dtype==jnp.float32 and new_state.m['p'].dtype==jnp.float32


@pytest.mark.parametrize('invalid',[float('nan'),float('inf')])
def test_true_nonfinite_gradients_still_reject_without_parameter_adam_updates(invalid):
    model=torch.nn.Linear(3,1,bias=False);optimizer=torch.optim.Adam(model.parameters())
    before=model.weight.detach().clone();model.weight.grad=torch.full_like(before,invalid)
    with pytest.raises(RuntimeError,match='non-finite'):
        clip_gradients_with_overflow_fallback(fabric(),model,optimizer,100.)
    assert not optimizer.state
    torch.testing.assert_close(model.weight,before)
    params={'p':jnp.asarray(before.numpy())};state=adam_init(params)
    (updated,new_state),_,valid=adam_update(params,state,
        {'p':jnp.full_like(params['p'],invalid)},1.,AdamSpec())
    assert not bool(valid) and int(new_state.count)==0
    np.testing.assert_array_equal(updated['p'],params['p'])
