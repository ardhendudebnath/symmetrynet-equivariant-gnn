"""The from-scratch layers compose into an exactly equivariant network.

This is the Phase 2 deliverable in test form: rotate the input, apply the corresponding
Wigner-D to the output, and confirm they agree to floating-point precision.
"""

from __future__ import annotations

import pytest
import torch

from symmetrynet.scratch.layer import ScratchTFN
from symmetrynet.scratch.spherical_harmonics import spherical_harmonics_l
from symmetrynet.scratch.tensor_product import EquivariantLinear, Gate, TensorProduct
from symmetrynet.scratch.wigner import random_rotation, wigner_D

DT = torch.float64
IRREPS_IN = {0: 4, 1: 4, 2: 2}
IRREPS_SH = {0: 1, 1: 1, 2: 1}
IRREPS_OUT = {0: 3, 1: 3, 2: 3}


def _rotate(feats: dict[int, torch.Tensor], rot: torch.Tensor) -> dict[int, torch.Tensor]:
    return {
        ell: torch.einsum("ij,nuj->nui", wigner_D(ell, rot), f) for ell, f in feats.items()
    }


@pytest.fixture
def features():
    return {ell: torch.randn(32, m, 2 * ell + 1, dtype=DT) for ell, m in IRREPS_IN.items()}


@pytest.mark.parametrize("ell", sorted(IRREPS_OUT))
def test_tensor_product_equivariance(features, ell: int):
    tp = TensorProduct(IRREPS_IN, IRREPS_SH, IRREPS_OUT).to(DT)
    vec = torch.randn(32, 3, dtype=DT)
    weights = torch.randn(32, tp.weight_numel, dtype=DT)
    rot = random_rotation(1, dtype=DT)

    sh = {k: spherical_harmonics_l(k, vec, normalization="component") for k in IRREPS_SH}
    sh_rot = {
        k: spherical_harmonics_l(k, vec @ rot.T, normalization="component") for k in IRREPS_SH
    }

    out = tp(features, sh, weights)
    out_rot = tp(_rotate(features, rot), sh_rot, weights)

    expected = torch.einsum("ij,nuj->nui", wigner_D(ell, rot), out[ell])
    assert torch.allclose(out_rot[ell], expected, atol=1e-11)


@pytest.mark.parametrize("ell", sorted(IRREPS_IN))
def test_equivariant_linear_equivariance(features, ell: int):
    lin = EquivariantLinear(IRREPS_IN, IRREPS_IN).to(DT)
    rot = random_rotation(1, dtype=DT)
    out_rot = lin(_rotate(features, rot))
    expected = torch.einsum("ij,nuj->nui", wigner_D(ell, rot), lin(features)[ell])
    assert torch.allclose(out_rot[ell], expected, atol=1e-11)


def test_equivariant_linear_has_no_bias_above_l0():
    """A bias on l>0 would break equivariance; it must not exist."""
    lin = EquivariantLinear(IRREPS_IN, IRREPS_IN)
    assert set(lin.biases.keys()) == {"0"}


@pytest.mark.parametrize("ell", sorted(IRREPS_IN))
def test_gate_equivariance(features, ell: int):
    gate = Gate(IRREPS_IN).to(DT)
    rot = random_rotation(1, dtype=DT)
    out_rot = gate(_rotate(features, rot))
    expected = torch.einsum("ij,nuj->nui", wigner_D(ell, rot), gate(features)[ell])
    assert torch.allclose(out_rot[ell], expected, atol=1e-11)


def test_gate_is_actually_nonlinear(features):
    """Guard against a 'gate' that silently degenerates into a linear map."""
    gate = Gate(IRREPS_IN).to(DT)
    single = gate(dict(features))[1]
    doubled = gate({ell: 2 * f for ell, f in features.items()})[1]
    assert not torch.allclose(doubled, 2 * single, atol=1e-6)


def test_tensor_product_rejects_impossible_irreps():
    with pytest.raises(ValueError, match="no CG path"):
        TensorProduct({0: 4}, {0: 1}, {2: 4})


@pytest.fixture
def model():
    return ScratchTFN(hidden_multiplicity=8, l_max=2, num_layers=2).to(DT).eval()


def test_model_rotation_invariance(model, molecules):
    """Predictions do not move under rotation -- for *any* weights, trained or not."""
    species, pos, batch, _ = molecules
    with torch.no_grad():
        base = model(species, pos, batch)
        for _ in range(25):
            rot = random_rotation(1, dtype=DT)
            assert torch.allclose(model(species, pos @ rot.T, batch), base, atol=1e-11)


def test_model_translation_invariance(model, molecules):
    species, pos, batch, _ = molecules
    with torch.no_grad():
        base = model(species, pos, batch)
        for _ in range(10):
            shift = torch.randn(3, dtype=DT) * 25.0
            assert torch.allclose(model(species, pos + shift, batch), base, atol=1e-10)


def test_model_is_o3_invariant(model, molecules):
    """Inversion too: every feature is a true tensor, so the l=0 readout is parity-even."""
    species, pos, batch, _ = molecules
    with torch.no_grad():
        base = model(species, pos, batch)
        for _ in range(10):
            improper = -random_rotation(1, dtype=DT)  # det = -1
            assert torch.allclose(model(species, pos @ improper.T, batch), base, atol=1e-11)


def test_model_permutation_invariance(model, molecules):
    species, pos, batch, sizes = molecules
    offsets = torch.tensor(sizes).cumsum(0) - torch.tensor(sizes)
    perm = torch.cat(
        [torch.randperm(n) + off for n, off in zip(sizes, offsets.tolist(), strict=True)]
    )
    with torch.no_grad():
        assert torch.allclose(
            model(species[perm], pos[perm], batch), model(species, pos, batch), atol=1e-11
        )


def test_model_is_not_trivially_constant(model, molecules):
    """Invariance is worthless if the model ignores its input entirely."""
    species, pos, batch, _ = molecules
    with torch.no_grad():
        base = model(species, pos, batch)
        perturbed = model(species, pos + torch.randn_like(pos) * 0.4, batch)
    assert (perturbed - base).abs().max() > 1e-6


def test_all_parameters_receive_gradient(molecules):
    """Every parameter must matter; a dead parameter usually means a wiring mistake."""
    species, pos, batch, _ = molecules
    model = ScratchTFN(hidden_multiplicity=8, l_max=2, num_layers=2)
    model(species, pos.float(), batch).sum().backward()
    dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not dead, f"parameters with no gradient: {dead}"


def test_gradients_wrt_positions_are_finite(molecules):
    species, pos, batch, _ = molecules
    model = ScratchTFN(hidden_multiplicity=8, l_max=2, num_layers=2)
    p = pos.float().requires_grad_(True)
    model(species, p, batch).sum().backward()
    assert torch.isfinite(p.grad).all()
