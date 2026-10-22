"""Density and force contracts; no OpenMM context or molecular simulation."""

import pytest

pytest.importorskip("normflows", reason="Requires the molecular extra: uv sync --extra molecular")

import math

import torch
from torch import nn

from tr_asbs.fab.boltzgen import distributions
from tr_asbs.fab.boltzgen.openmm_interface import apply_energy_mode, regularize_energy
from tr_asbs.potential.internal_coordinate_chart import InternalCoordinateChart


class FakePool:
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def join(self):
        pass


CLASSES = ('Boltzmann', 'TransformedBoltzmann', 'BoltzmannParallel',
           'TransformedBoltzmannParallel')


def distribution(monkeypatch, name, mode):
    monkeypatch.setattr(distributions.mp, 'Pool', FakePool)
    arguments = dict(temperature=300, energy_cut=3., energy_max=20., energy_mode=mode)
    parallel = 'Parallel' in name
    arguments['system' if parallel else 'sim_context'] = None
    if parallel:
        arguments['n_threads'] = 1
    if name.startswith('Transformed'):
        arguments['transform'] = lambda z: (2 * z, z.new_full((len(z),), math.log(2)))
    result = getattr(distributions, name)(**arguments)
    def raw(x, *_):
        return x.square().sum(-1, keepdim=True)
    result.openmm_energy = (lambda x, *_: (raw(x), 2 * x)) if parallel else raw
    return result


@pytest.mark.parametrize('name', CLASSES)
@pytest.mark.parametrize('mode', ('regularized', 'physical'))
def test_modes_preserve_target_jacobian_and_correct_gradient(monkeypatch, name, mode):
    target = distribution(monkeypatch, name, mode)
    y = torch.tensor([[2.], [10.]], dtype=torch.double, requires_grad=True)
    transformed = name.startswith('Transformed')
    cart = 2 * y if transformed else y
    raw = cart.square().sum(-1)
    expected_energy = (raw if mode == 'physical' else
                       regularize_energy(raw, torch.tensor(3.), torch.tensor(20.)))
    expected = -expected_energy + (math.log(2) if transformed else 0)
    actual = target.log_prob(y)
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.sum(), y, retain_graph=True)[0]
    torch.testing.assert_close(actual_gradient, torch.autograd.grad(expected.sum(), y)[0])
    if mode == 'physical':
        torch.testing.assert_close(actual_gradient, -2 * y * (4 if transformed else 1))
    else:
        assert actual_gradient[-1].item() == 0  # finite legacy tail cap


@pytest.mark.parametrize('name', CLASSES)
def test_invalid_mode_rejected_before_backend_setup(monkeypatch, name):
    with pytest.raises(ValueError, match='energy_mode'):
        distribution(monkeypatch, name, 'typo')


def test_physical_mode_keeps_nonfinite_values_visible():
    values = torch.tensor([float('nan'), float('inf')])
    result = apply_energy_mode(values, torch.tensor(3.), torch.tensor(20.), 'physical')
    assert result is values
    assert torch.isnan(result[0]) and torch.isposinf(result[1])


def test_unbounded_bond_chart_does_not_restore_capped_density_decay():
    chart = InternalCoordinateChart.__new__(InternalCoordinateChart)
    nn.Module.__init__(chart)
    chart._initialize(torch.ones(1, dtype=torch.double), torch.ones(1, dtype=torch.double),
                      torch.zeros(1, dtype=torch.long))
    z, chart_jac = chart(torch.tensor([[100.], [1000.]], dtype=torch.double))
    bond = z[:, 0] + 1
    energy = bond.square()
    # Cartesian IC reconstruction contributes b^2 db at fixed nondegenerate angle.
    jacobian = 2 * torch.log(bond) + chart_jac
    capped = -regularize_energy(energy, torch.tensor(3.), torch.tensor(20.)) + jacobian
    physical = -energy + jacobian
    assert capped[1] > capped[0]
    assert physical[1] < physical[0]


def test_calibration_energy_banks_cannot_cross_density_definitions():
    from tr_asbs.potential.alanine_dipeptide_energy import validate_calibration_energy_definition
    physical = dict(mode='physical', temperature=300)
    regularized = dict(mode='regularized', temperature=300, energy_cut=1e8, energy_max=1e20)
    validate_calibration_energy_definition({}, regularized)  # historical bank compatibility
    validate_calibration_energy_definition(dict(energy_definition=physical), physical)
    with pytest.raises(ValueError, match='versioned'):
        validate_calibration_energy_definition({}, physical)
    with pytest.raises(ValueError, match='definition mismatch'):
        validate_calibration_energy_definition(dict(energy_definition=regularized), physical)
    with pytest.raises(ValueError, match='definition mismatch'):
        validate_calibration_energy_definition(dict(energy_definition=physical), regularized)
