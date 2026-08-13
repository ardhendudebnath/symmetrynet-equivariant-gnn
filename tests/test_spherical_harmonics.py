"""The from-scratch spherical harmonics agree with e3nn and with the theory."""

from __future__ import annotations

import math

import pytest
import torch
from e3nn import o3

from symmetrynet.scratch.spherical_harmonics import (
    MAX_L,
    e3nn_to_standard_frame,
    spherical_harmonics,
    spherical_harmonics_l,
    standard_to_e3nn_frame,
)
from symmetrynet.scratch.wigner import random_rotation, wigner_D, wigner_D_blocks

DEGREES = list(range(MAX_L + 1))
DT = torch.float64


@pytest.mark.parametrize("ell", DEGREES)
def test_matches_e3nn_after_frame_relabelling(ell: int):
    """Ours equals e3nn's once the differing axis convention is accounted for.

    e3nn treats its *second* coordinate as the polar axis; we use the standard
    physics convention with z polar.  Under the documented relabelling the two
    implementations agree to machine precision, which pins the convention down
    instead of leaving it as folklore.
    """
    v_e3nn = torch.randn(128, 3, dtype=DT)
    mine = spherical_harmonics_l(ell, e3nn_to_standard_frame(v_e3nn))
    theirs = o3.spherical_harmonics(ell, v_e3nn, normalize=True, normalization="integral")
    assert torch.allclose(mine, theirs, atol=1e-12)


def test_frame_conversions_are_inverse():
    v = torch.randn(64, 3, dtype=DT)
    assert torch.allclose(standard_to_e3nn_frame(e3nn_to_standard_frame(v)), v)


@pytest.mark.parametrize("ell", DEGREES)
def test_addition_theorem(ell: int):
    r""":math:`\sum_m Y_{\ell m}(\hat r)^2 = (2\ell+1)/4\pi` for any unit vector."""
    y = spherical_harmonics_l(ell, torch.randn(256, 3, dtype=DT))
    expected = (2 * ell + 1) / (4 * math.pi)
    assert torch.allclose(y.pow(2).sum(-1), torch.full((256,), expected, dtype=DT))


@pytest.mark.parametrize("normalization", ["integral", "component", "norm"])
@pytest.mark.parametrize("ell", DEGREES)
def test_normalization_conventions(ell: int, normalization: str):
    y = spherical_harmonics_l(ell, torch.randn(64, 3, dtype=DT), normalization=normalization)
    sq = y.pow(2).sum(-1)
    expected = {
        "integral": (2 * ell + 1) / (4 * math.pi),
        "component": float(2 * ell + 1),
        "norm": 1.0,
    }[normalization]
    assert torch.allclose(sq, torch.full((64,), expected, dtype=DT))


@pytest.mark.parametrize("ell", DEGREES)
def test_equivariance_under_rotation(ell: int):
    """The defining property: ``Y(Rr) == D(R) Y(r)``.

    This is *the* test.  The Wigner-D matrix is built algebraically from Clebsch-Gordan
    coefficients and never sees a spherical harmonic, so agreement here is a genuine
    cross-check of two independent derivations rather than a tautology.
    """
    vecs = torch.randn(256, 3, dtype=DT)
    rot = random_rotation(1, dtype=DT)
    rotated = spherical_harmonics_l(ell, vecs @ rot.T)
    transformed = spherical_harmonics_l(ell, vecs) @ wigner_D(ell, rot).T
    assert torch.allclose(rotated, transformed, atol=1e-12)


def test_stacked_degrees_equivariance():
    vecs = torch.randn(128, 3, dtype=DT)
    rot = random_rotation(1, dtype=DT)
    rotated = spherical_harmonics(DEGREES, vecs @ rot.T)
    transformed = spherical_harmonics(DEGREES, vecs) @ wigner_D_blocks(DEGREES, rot).T
    assert torch.allclose(rotated, transformed, atol=1e-12)


@pytest.mark.parametrize("ell", DEGREES)
def test_parity(ell: int):
    r""":math:`Y_\ell(-\hat r) = (-1)^{\ell} Y_\ell(\hat r)`.

    This is why odd degrees behave as pseudo-vectors under inversion, and it is what
    fixes the parity labels (``0e``, ``1o``, ``2e``, ...) used by the e3nn model.
    """
    vecs = torch.randn(64, 3, dtype=DT)
    assert torch.allclose(
        spherical_harmonics_l(ell, -vecs),
        ((-1.0) ** ell) * spherical_harmonics_l(ell, vecs),
        atol=1e-12,
    )


def test_l1_is_the_direction_vector_reordered():
    """Sanity anchor: l=1 is literally the unit vector in (y, z, x) order."""
    vecs = torch.randn(32, 3, dtype=DT)
    unit = vecs / vecs.norm(dim=-1, keepdim=True)
    y1 = spherical_harmonics_l(1, vecs, normalization="norm")
    assert torch.allclose(y1, unit[:, [1, 2, 0]], atol=1e-12)


def test_zero_vector_does_not_produce_nan():
    """Degenerate input must not poison a whole training batch with NaNs."""
    y = spherical_harmonics(DEGREES, torch.zeros(4, 3, dtype=DT))
    assert torch.isfinite(y).all()


def test_rejects_degree_beyond_closed_forms():
    with pytest.raises(ValueError, match="closed forms"):
        spherical_harmonics_l(MAX_L + 1, torch.randn(4, 3, dtype=DT))
