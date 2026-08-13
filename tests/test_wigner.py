"""Wigner-D matrices form a representation of SO(3)."""

from __future__ import annotations

import math

import pytest
import torch

from symmetrynet.scratch.wigner import (
    random_rotation,
    rotation_matrix,
    wigner_D,
    wigner_D_blocks,
)

DT = torch.float64
DEGREES = list(range(4))


@pytest.mark.parametrize("ell", DEGREES)
def test_orthogonal(ell: int):
    d = wigner_D(ell, random_rotation(1, dtype=DT))
    assert torch.allclose(d @ d.T, torch.eye(2 * ell + 1, dtype=DT), atol=1e-12)


@pytest.mark.parametrize("ell", DEGREES)
def test_group_homomorphism(ell: int):
    """``D(R1 R2) == D(R1) D(R2)`` -- the defining property of a representation."""
    r1, r2 = random_rotation(1, dtype=DT), random_rotation(1, dtype=DT)
    assert torch.allclose(wigner_D(ell, r1 @ r2), wigner_D(ell, r1) @ wigner_D(ell, r2), atol=1e-12)


@pytest.mark.parametrize("ell", DEGREES)
def test_identity_maps_to_identity(ell: int):
    d = wigner_D(ell, torch.eye(3, dtype=DT))
    assert torch.allclose(d, torch.eye(2 * ell + 1, dtype=DT), atol=1e-12)


@pytest.mark.parametrize("ell", DEGREES)
def test_inverse(ell: int):
    r = random_rotation(1, dtype=DT)
    assert torch.allclose(wigner_D(ell, r.T), wigner_D(ell, r).T, atol=1e-12)


@pytest.mark.parametrize("ell", DEGREES)
def test_determinant_is_one(ell: int):
    d = wigner_D(ell, random_rotation(1, dtype=DT))
    assert torch.det(d).item() == pytest.approx(1.0, abs=1e-10)


def test_l1_is_the_rotation_matrix_reordered():
    """``D^1`` is nothing but ``R`` rewritten in the (y, z, x) basis."""
    r = random_rotation(1, dtype=DT)
    perm = [1, 2, 0]
    assert torch.allclose(wigner_D(1, r), r[perm][:, perm], atol=1e-12)


def test_batched():
    rots = random_rotation(16, dtype=DT)
    d = wigner_D(2, rots)
    assert d.shape == (16, 5, 5)
    assert torch.allclose(d[3], wigner_D(2, rots[3]), atol=1e-12)


def test_rotation_matrix_is_orthogonal_with_unit_determinant():
    a, b, g = (torch.rand(5, dtype=DT) * math.tau for _ in range(3))
    r = rotation_matrix(a, b, g)
    identity = torch.eye(3, dtype=DT).expand(5, 3, 3)
    assert torch.allclose(r @ r.transpose(-1, -2), identity, atol=1e-12)
    assert torch.allclose(torch.det(r), torch.ones(5, dtype=DT), atol=1e-12)


def test_random_rotation_is_haar_uniform():
    """A rotated fixed vector must cover the sphere uniformly.

    Sampling all three Euler angles uniformly would cluster points near the poles.
    Checking that the mean of ``R z`` is ~0 and that ``cos(theta)`` is uniform catches
    that mistake, which is otherwise invisible until equivariance results look odd.
    """
    rots = random_rotation(20000, dtype=DT, generator=torch.Generator().manual_seed(0))
    pts = rots @ torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    assert pts.mean(0).norm().item() < 0.05
    cos_theta = pts[:, 2]
    # Uniform on [-1, 1] has mean 0 and variance 1/3.
    assert abs(cos_theta.mean().item()) < 0.05
    assert abs(cos_theta.var().item() - 1 / 3) < 0.05


def test_blocks_are_block_diagonal():
    degrees = [0, 1, 2]
    d = wigner_D_blocks(degrees, random_rotation(1, dtype=DT))
    assert d.shape == (9, 9)
    assert torch.allclose(d[0:1, 1:], torch.zeros(1, 8, dtype=DT), atol=1e-14)
    assert torch.allclose(d[1:4, 4:], torch.zeros(3, 5, dtype=DT), atol=1e-14)


def test_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        wigner_D(1, torch.randn(4, 2, dtype=DT))
