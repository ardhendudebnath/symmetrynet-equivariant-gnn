"""Clebsch-Gordan coefficients: orthonormality, selection rules, and intertwining."""

from __future__ import annotations

import itertools

import pytest
import torch
from e3nn import o3

from symmetrynet.scratch.clebsch_gordan import (
    clebsch_gordan,
    clebsch_gordan_element,
    decomposition_degrees,
    real_basis_change,
    wigner_3j,
)
from symmetrynet.scratch.wigner import random_rotation, wigner_D

DT = torch.float64
PATHS = [
    (l1, l2, l3)
    for l1, l2 in itertools.product(range(3), repeat=2)
    for l3 in decomposition_degrees(l1, l2)
    if l3 <= 3
]


def test_known_coefficient_values():
    """Spot-check the Racah formula against textbook values."""
    assert clebsch_gordan_element(1, 0, 1, 0, 0, 0) == pytest.approx(-(3**-0.5))
    assert clebsch_gordan_element(1, 1, 1, -1, 0, 0) == pytest.approx(3**-0.5)
    assert clebsch_gordan_element(1, 1, 1, 1, 2, 2) == pytest.approx(1.0)
    # Selection rules
    assert clebsch_gordan_element(1, 1, 1, 0, 0, 0) == 0.0  # m != m1 + m2
    assert clebsch_gordan_element(1, 0, 1, 0, 3, 0) == 0.0  # l outside |l1-l2|..l1+l2


def test_selection_rule():
    assert decomposition_degrees(1, 1) == [0, 1, 2]
    assert decomposition_degrees(2, 1) == [1, 2, 3]
    assert decomposition_degrees(0, 0) == [0]
    assert decomposition_degrees(2, 2, l_max=2) == [0, 1, 2]


@pytest.mark.parametrize("l1,l2,l3", PATHS)
def test_columns_orthonormal(l1: int, l2: int, l3: int):
    """``C^T C = I`` -- the convention the Wigner-D recursion depends on."""
    mat = clebsch_gordan(l1, l2, l3, dtype=DT).reshape(-1, 2 * l3 + 1)
    assert torch.allclose(mat.T @ mat, torch.eye(2 * l3 + 1, dtype=DT), atol=1e-12)


@pytest.mark.parametrize("l1,l2,l3", PATHS)
def test_intertwines_representations(l1: int, l2: int, l3: int):
    r"""``(D^l1 (x) D^l2) C == C D^l3`` -- the identity that makes equivariance work."""
    rot = random_rotation(8, dtype=DT)
    cg = clebsch_gordan(l1, l2, l3, dtype=DT).reshape(-1, 2 * l3 + 1)
    kron = torch.einsum("bik,bjl->bijkl", wigner_D(l1, rot), wigner_D(l2, rot)).reshape(
        rot.shape[0], (2 * l1 + 1) * (2 * l2 + 1), -1
    )
    assert torch.allclose(kron @ cg, cg @ wigner_D(l3, rot), atol=1e-12)


@pytest.mark.parametrize("l1,l2,l3", PATHS)
def test_is_real(l1: int, l2: int, l3: int):
    """The real-basis coefficients must be genuinely real, not complex with small imag."""
    cg = clebsch_gordan(l1, l2, l3, dtype=DT)
    assert not cg.is_complex()
    assert torch.isfinite(cg).all()


@pytest.mark.parametrize("l1,l2,l3", PATHS)
def test_wigner_3j_matches_e3nn(l1: int, l2: int, l3: int):
    """Matches e3nn up to an overall sign, which is pure convention.

    Any orthonormal intertwiner is defined only up to a global phase; both libraries
    pick one arbitrarily, and the learned weights absorb it either way.  The tolerance
    is 1e-6 because e3nn stores these in float32.
    """
    mine = wigner_3j(l1, l2, l3, dtype=DT)
    theirs = o3.wigner_3j(l1, l2, l3).to(DT)
    err = min((mine - theirs).abs().max(), (mine + theirs).abs().max())
    assert err < 1e-6


@pytest.mark.parametrize("ell", range(4))
def test_real_basis_change_is_unitary(ell: int):
    u = real_basis_change(ell)
    assert torch.allclose(
        u @ u.conj().T, torch.eye(2 * ell + 1, dtype=torch.complex128), atol=1e-12
    )


def test_scalar_path_is_the_dot_product():
    """``1 (x) 1 -> 0`` must reproduce the normalised dot product.

    The overall sign is convention (we fix it by forcing the first non-zero entry
    positive), so the meaningful assertion is that the tensor is a multiple of the
    identity with magnitude ``1/sqrt(3)`` -- i.e. contracting two vectors through it
    gives their dot product up to normalisation.
    """
    cg = clebsch_gordan(1, 1, 0, dtype=DT).squeeze(-1)
    assert torch.allclose(cg.abs(), torch.eye(3, dtype=DT) / 3**0.5, atol=1e-12)

    u, v = torch.randn(2, 3, dtype=DT)
    got = torch.einsum("ij,i,j->", cg, u, v)
    assert abs(abs(got.item()) - abs(torch.dot(u, v).item()) / 3**0.5) < 1e-12


def test_vector_path_is_the_cross_product():
    """``1 (x) 1 -> 1`` must be proportional to the Levi-Civita tensor (cross product)."""
    cg = clebsch_gordan(1, 1, 1, dtype=DT)
    u, v = torch.randn(2, 3, dtype=DT)
    got = torch.einsum("ijk,i,j->k", cg, u, v)
    # Cross product in the (y, z, x) ordering the real harmonics use.
    cross = torch.linalg.cross(u[[2, 0, 1]], v[[2, 0, 1]])[[1, 2, 0]]
    ratio = got / cross
    assert torch.allclose(ratio, ratio[0].expand(3), atol=1e-10)


def test_invalid_path_raises():
    with pytest.raises(ValueError, match="selection rule"):
        clebsch_gordan(1, 1, 3)
