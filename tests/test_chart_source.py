"""Shifted-chart Jacobians and physically matched sources, without OpenMM."""

import math

import torch
from torch import nn

from tr_asbs.potential.internal_coordinate_chart import InternalCoordinateChart
from tr_asbs.process.chart_source import ChartMappedGaussianPrior


def test_shifted_chart_and_source_change_of_measure(tmp_path):
    base = InternalCoordinateChart.__new__(InternalCoordinateChart)
    nn.Module.__init__(base)
    base._initialize(torch.tensor([2., 1.2, .4], dtype=torch.double),
                     torch.tensor([.3, .2, .6], dtype=torch.double), torch.tensor([0, 1, 2]))
    shift = torch.tensor([0., 0., 1.1], dtype=torch.double)
    shifted = InternalCoordinateChart.from_state(base.state_dict(), shift)
    bundle = tmp_path / 'calibration.pt'
    torch.save(dict(schema='aldp_chart_calibration_v1', base_chart=base.state_dict(),
                    chart_offsets={'base': torch.zeros(3), 'shift': shift}), bundle)
    p = ChartMappedGaussianPrior(bundle, 'base', dim=3)
    q = ChartMappedGaussianPrior(bundle, 'shift', dim=3)
    torch.manual_seed(3)
    a = p.sample((8,))
    torch.manual_seed(3)
    b = q.sample((8,))
    za, ja = base(a)
    zb, jb = shifted(b)
    torch.testing.assert_close(base.canonicalize(za), shifted.canonicalize(zb))
    torch.testing.assert_close(p.log_prob(a) - ja, q.log_prob(b) - jb, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(shifted.inverse(zb), b)
    torch.testing.assert_close(shifted.inverse(base.canonicalize(zb), canonical=True), b)
    value = b[:1].double().requires_grad_(True)
    matrix = torch.autograd.functional.jacobian(
        lambda v: shifted(v[None])[0][0], value[0])
    torch.testing.assert_close(shifted(value)[1][0], torch.linalg.slogdet(matrix).logabsdet)
    assert torch.isfinite(q.score(value)).all()


def test_circle_cut_changes_coordinates_but_not_periodic_physical_value():
    chart = InternalCoordinateChart.__new__(InternalCoordinateChart)
    nn.Module.__init__(chart)
    chart._initialize(torch.tensor([0.]), torch.tensor([1.]), torch.tensor([2]))
    chart.set_dihedral_offsets(torch.tensor([1.]))
    z = torch.tensor([[-2.5], [2.5]])
    y = chart.inverse(z, canonical=True)
    restored, _ = chart(y)
    torch.testing.assert_close(torch.sin(restored), torch.sin(z), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.cos(restored), torch.cos(z), atol=1e-6, rtol=1e-6)
    assert (restored > 1 - math.pi).all() and (restored < 1 + math.pi).all()
