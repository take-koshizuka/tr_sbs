"""Bijective IC support and Jacobian verification without invoking OpenMM."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from tr_asbs.potential.internal_coordinate_chart import InternalCoordinateChart
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_aldp_chart_campaign import cartesian_roundtrip_diagnostic


def chart():
    result = InternalCoordinateChart.__new__(InternalCoordinateChart)
    nn.Module.__init__(result)
    result._initialize(
        torch.tensor([2.0, 1.2, 0.4], dtype=torch.double),
        torch.tensor([0.3, 0.2, 0.6], dtype=torch.double),
        torch.tensor([0, 1, 2]),
    )
    return result


def test_chart_roundtrip_support_and_local_scaling():
    transform = chart()
    y = torch.tensor(
        [[-4.0, -3.0, -2.0], [0.0, 0.0, 0.0], [2.0, 3.0, 4.0]], dtype=torch.double
    )
    z, jac = transform(y)
    physical = z * transform.scales + transform.means
    assert (physical[:, 0] > 0).all()
    assert ((physical[:, 1] > 0) & (physical[:, 1] < math.pi)).all()
    assert ((z[:, 2] * transform.scales[2]).abs() < math.pi).all()
    torch.testing.assert_close(transform.inverse(z), y)
    torch.testing.assert_close(
        z[1], torch.zeros(3, dtype=torch.double), atol=1e-15, rtol=0
    )
    assert torch.isfinite(jac).all()


def test_reported_jacobian_matches_autograd_and_change_of_measure():
    transform = chart()
    y = torch.tensor([[0.4, -0.3, 0.7]], dtype=torch.double, requires_grad=True)
    z, logjac = transform(y)
    matrix = torch.autograd.functional.jacobian(
        lambda v: transform(v[None])[0][0], y[0]
    )
    torch.testing.assert_close(logjac[0], torch.linalg.slogdet(matrix).logabsdet)

    # Finite differences check the potential-gradient sign and all Jacobian terms.
    def energy(v):
        zv, jv = transform(v)
        return 0.5 * zv.square().sum(-1) - jv

    gradient = torch.autograd.grad(energy(y).sum(), y)[0]
    finite = torch.zeros_like(y)
    for index in range(3):
        delta = torch.zeros_like(y)
        delta[0, index] = 1e-5
        finite[0, index] = (energy(y + delta) - energy(y - delta)) / (2e-5)
    torch.testing.assert_close(gradient, finite, atol=1e-8, rtol=1e-7)


def test_inverse_rejects_multiple_periods_and_invalid_bonds():
    transform = chart()
    with pytest.raises(ValueError, match="dihedral"):
        transform.inverse(
            torch.tensor([[0.0, 0.0, 2 * math.pi / 0.6]], dtype=torch.double)
        )
    with pytest.raises(ValueError, match="bond"):
        transform.inverse(torch.tensor([[-10.0, 0.0, 0.0]], dtype=torch.double))


def diagnostic_potential(transform):
    return SimpleNamespace(chart=transform,native_to_cartesian=lambda y:transform(y)[0],
        cartesian_to_native=lambda x:transform.inverse(x,canonical=True))


def test_roundtrip_diagnostic_preserves_samples_and_rng():
    transform=chart()
    y=torch.tensor([[.2,-.4,.7]],dtype=torch.double)
    x=transform(y)[0]
    original_y,original_x=y.clone(),x.clone()
    state=torch.get_rng_state().clone()
    report=cartesian_roundtrip_diagnostic(diagnostic_potential(transform),y,x)
    assert report['status']=='passed' and report['error_nm']<2e-5
    assert report['forward_chart_boundary_sample_count']==0
    assert report['additional_openmm_configurations']==0
    assert not report['samples_modified'] and not report['samples_filtered']
    assert torch.equal(state,torch.get_rng_state())
    assert torch.equal(y,original_y) and torch.equal(x,original_x)


@pytest.mark.parametrize('dimension,error_name',[(1,'angle'),(2,'dihedral')])
def test_float32_saturation_is_retained_without_relaxing_strict_inverse(dimension,error_name):
    transform=chart().float()
    y=torch.zeros(3,3)
    # These endpoints reproduce rejection in this float32 calibration. The
    # opposite angle endpoint can round slightly inside the strict support.
    y[:2,dimension]=torch.tensor([-1000.,-1000.] if dimension==1 else [-1000.,1000.])
    x=transform(y)[0]
    with pytest.raises(ValueError,match=error_name):
        transform.inverse(x,canonical=True)
    before=y.clone(),x.clone()
    report=cartesian_roundtrip_diagnostic(diagnostic_potential(transform),y,x)
    assert report['status']=='strict_inverse_rejected' and report['error_nm'] is None
    assert report['sample_count']==3 and error_name in report['inverse_error']
    assert report['forward_chart_boundary_sample_count']==2
    assert report['forward_chart_boundary_samples'][error_name]==2
    assert torch.equal(y,before[0]) and torch.equal(x,before[1])
    with pytest.raises(ValueError,match=error_name):
        transform.inverse(x,canonical=True)


def test_roundtrip_diagnostic_does_not_hide_unexpected_errors():
    transform=chart()
    potential=diagnostic_potential(transform)
    def broken(_):
        raise RuntimeError('unexpected transform failure')
    potential.cartesian_to_native=broken
    y=torch.zeros(1,3,dtype=torch.double)
    with pytest.raises(RuntimeError,match='unexpected transform'):
        cartesian_roundtrip_diagnostic(potential,y,transform(y)[0])
