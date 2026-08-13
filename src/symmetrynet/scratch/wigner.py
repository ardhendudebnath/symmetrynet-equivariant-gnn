r"""Wigner-D matrices for the real spherical-harmonic basis, plus rotation sampling.

The Wigner-D matrix :math:`D^{\ell}(R)` is *the* concrete answer to "what does a rotation
do to a degree-:math:`\ell` feature".  It is defined by

.. math::  Y_{\ell}(R\,\hat r) \;=\; D^{\ell}(R)\, Y_{\ell}(\hat r),

and it is what turns the abstract statement "this feature lives in irrep :math:`\ell`"
into something a test can check numerically.

Construction (no library calls, no circularity)
-----------------------------------------------
It would be circular to *fit* :math:`D^{\ell}` from spherical harmonics and then use it
to test those same harmonics.  So we build it algebraically instead:

* :math:`D^{0}(R) = [1]` -- scalars do not move.
* :math:`D^{1}(R) = P R P^{\top}`, where :math:`P` is the permutation taking
  :math:`(x, y, z)` to :math:`(y, z, x)`.  This is just the rotation matrix itself,
  rewritten in the :math:`m = -1, 0, +1` ordering the real harmonics use.
* :math:`D^{\ell}(R) = C^{\top} \bigl(D^{\ell-1}(R) \otimes D^{1}(R)\bigr) C` for
  :math:`\ell \ge 2`, where :math:`C` is the Clebsch-Gordan intertwiner for
  :math:`(\ell-1) \otimes 1 \to \ell`.

The last line is the intertwining identity read backwards, and it is valid precisely
because ``C`` has orthonormal columns.  So the only inputs are the rotation matrix and
the CG coefficients derived in :mod:`symmetrynet.scratch.clebsch_gordan` -- the
spherical harmonics play no part, which is what makes the equivariance test meaningful.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .clebsch_gordan import clebsch_gordan

__all__ = [
    "rotation_matrix",
    "random_rotation",
    "random_reflection",
    "wigner_D",
    "wigner_D_blocks",
]

# Permutation carrying an (x, y, z)-ordered vector into the (y, z, x) order that the
# real harmonics use for m = -1, 0, +1.
_P_YZX = torch.tensor(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64
)


def _rot_z(angle: Tensor) -> Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    o, z = torch.ones_like(c), torch.zeros_like(c)
    return torch.stack(
        [
            torch.stack([c, -s, z], dim=-1),
            torch.stack([s, c, z], dim=-1),
            torch.stack([z, z, o], dim=-1),
        ],
        dim=-2,
    )


def _rot_y(angle: Tensor) -> Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    o, z = torch.ones_like(c), torch.zeros_like(c)
    return torch.stack(
        [
            torch.stack([c, z, s], dim=-1),
            torch.stack([z, o, z], dim=-1),
            torch.stack([-s, z, c], dim=-1),
        ],
        dim=-2,
    )


def rotation_matrix(alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
    r"""Rotation from ZYZ Euler angles: :math:`R = R_z(\alpha) R_y(\beta) R_z(\gamma)`."""
    return _rot_z(alpha) @ _rot_y(beta) @ _rot_z(gamma)


def random_rotation(
    n: int = 1,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Sample ``n`` rotations from the Haar (uniform) measure on SO(3).

    Sampling all three Euler angles uniformly would **not** be uniform on SO(3) -- it
    clusters near the poles.  The correct measure needs
    :math:`\cos\beta \sim \mathcal{U}(-1, 1)`, which is what we do here.

    Returns shape ``(n, 3, 3)``, or ``(3, 3)`` when ``n == 1``.
    """
    kw = {"dtype": dtype, "device": device, "generator": generator}
    alpha = torch.rand(n, **kw) * (2 * math.pi)
    beta = torch.acos(2.0 * torch.rand(n, **kw) - 1.0)
    gamma = torch.rand(n, **kw) * (2 * math.pi)
    out = rotation_matrix(alpha, beta, gamma)
    return out[0] if n == 1 else out


def random_reflection(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """A random improper orthogonal matrix (``det = -1``), for testing O(3) parity."""
    r = random_rotation(1, dtype=dtype, device=device, generator=generator)
    return -r  # negating a 3x3 rotation flips the determinant sign


def _batched_kron(a: Tensor, b: Tensor) -> Tensor:
    """Kronecker product over trailing 2 dims, broadcasting the leading batch dims."""
    p, q = a.shape[-1], b.shape[-1]
    out = torch.einsum("...ik,...jl->...ijkl", a, b)
    return out.reshape(*a.shape[:-2], p * q, p * q)


def wigner_D(ell: int, rotation: Tensor) -> Tensor:
    r"""Wigner-D matrix for degree ``ell``.

    Parameters
    ----------
    ell:
        Non-negative degree.
    rotation:
        Orthogonal matrices of shape ``(..., 3, 3)``.  Proper rotations give the
        genuine :math:`D^{\ell}`; passing an improper matrix returns the O(3) action
        on a *pseudo*-tensor, which for even ``ell`` equals the action of ``-R``.

    Returns
    -------
    Tensor of shape ``(..., 2*ell+1, 2*ell+1)``.
    """
    if ell < 0:
        raise ValueError(f"degree must be non-negative, got {ell}")
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"expected trailing shape (3, 3), got {tuple(rotation.shape)}")

    dtype, device = rotation.dtype, rotation.device
    batch = rotation.shape[:-2]

    if ell == 0:
        return torch.ones(*batch, 1, 1, dtype=dtype, device=device)

    p = _P_YZX.to(dtype=dtype, device=device)
    d1 = p @ rotation @ p.transpose(-1, -2)
    if ell == 1:
        return d1

    d = d1
    for k in range(2, ell + 1):
        cg = clebsch_gordan(k - 1, 1, k, dtype=dtype, device=device)
        cg = cg.reshape((2 * (k - 1) + 1) * 3, 2 * k + 1)
        kron = _batched_kron(d, d1)
        d = torch.einsum("ia,...ij,jb->...ab", cg, kron, cg)
    return d


def wigner_D_blocks(ells: list[int], rotation: Tensor) -> Tensor:
    r"""Block-diagonal :math:`\bigoplus_\ell D^{\ell}(R)` matching a concatenated feature.

    Handy for testing a stacked harmonic vector such as the ``[0, 1, 2]`` output of
    :func:`symmetrynet.scratch.spherical_harmonics.spherical_harmonics`.
    """
    blocks = [wigner_D(ell, rotation) for ell in ells]
    total = sum(2 * ell + 1 for ell in ells)
    batch = rotation.shape[:-2]
    out = torch.zeros(*batch, total, total, dtype=rotation.dtype, device=rotation.device)
    off = 0
    for ell, block in zip(ells, blocks, strict=True):
        d = 2 * ell + 1
        out[..., off : off + d, off : off + d] = block
        off += d
    return out
