r"""Real spherical harmonics, written out in closed form.

This module is deliberately *not* a call into ``e3nn``.  Phase 2 of the project is a
proof of understanding: everything here is derived by hand from the standard real
spherical harmonics and validated numerically against ``e3nn`` in
``tests/test_spherical_harmonics.py``.

Conventions
-----------
We use the **standard physics convention**:

* the polar axis is :math:`z`;
* for degree :math:`\ell` the :math:`2\ell+1` components are ordered
  :math:`m = -\ell, \dots, 0, \dots, +\ell`;
* the normalisation is *integral*, i.e. :math:`\int_{S^2} |Y_{\ell m}|^2 \, d\Omega = 1`.

Two consequences are worth internalising because they trip people up constantly.

1. Under this ordering the :math:`\ell=1` harmonics are proportional to
   :math:`(y, z, x)` -- **not** :math:`(x, y, z)`.  That is not a typo: the real
   harmonic with :math:`m=-1` is :math:`\propto y`, with :math:`m=0` is
   :math:`\propto z`, and with :math:`m=+1` is :math:`\propto x`.

2. ``e3nn`` uses these same functions but relabels the axes so that *its* second
   coordinate is the polar axis.  Concretely, for an input vector ``v`` expressed in
   e3nn's frame, e3nn's output equals ours evaluated at ``(v[2], v[0], v[1])``.
   :func:`e3nn_to_standard_frame` performs that relabelling, and the test suite pins
   the equivalence down to floating-point precision.

The closed forms below are the textbook real solid harmonics restricted to the unit
sphere (so :math:`r = 1` and e.g. :math:`3z^2 - r^2 \to 3z^2 - 1`).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "MAX_L",
    "spherical_harmonics",
    "spherical_harmonics_l",
    "e3nn_to_standard_frame",
    "standard_to_e3nn_frame",
    "normalization_factor",
]

#: Highest degree written out in closed form here.  Going beyond this is what the
#: e3nn-based Phase 3 model is for; by hand it stops being instructive.
MAX_L = 3


def normalization_factor(ell: int, normalization: str) -> float:
    r"""Scale converting *integral*-normalised harmonics to another convention.

    On the unit sphere the addition theorem gives
    :math:`\sum_m Y_{\ell m}(\hat r)^2 = (2\ell+1) / 4\pi`, so:

    * ``"integral"`` -- :math:`\int |Y|^2 d\Omega = 1` (factor 1);
    * ``"component"`` -- :math:`\sum_m Y_{\ell m}^2 = 2\ell + 1` (factor :math:`\sqrt{4\pi}`);
    * ``"norm"`` -- :math:`\sum_m Y_{\ell m}^2 = 1` (factor :math:`\sqrt{4\pi/(2\ell+1)}`).
    """
    if normalization == "integral":
        return 1.0
    if normalization == "component":
        return math.sqrt(4.0 * math.pi)
    if normalization == "norm":
        return math.sqrt(4.0 * math.pi / (2 * ell + 1))
    raise ValueError(
        f"unknown normalization {normalization!r}; expected 'integral', 'component' or 'norm'"
    )


def e3nn_to_standard_frame(v: Tensor) -> Tensor:
    """Map a vector from e3nn's axis convention into ours.

    e3nn treats its *second* coordinate as the polar axis, so ``(x, y, z)_standard``
    corresponds to ``(v[..., 2], v[..., 0], v[..., 1])`` in e3nn's frame.
    """
    return torch.stack((v[..., 2], v[..., 0], v[..., 1]), dim=-1)


def standard_to_e3nn_frame(v: Tensor) -> Tensor:
    """Inverse of :func:`e3nn_to_standard_frame`."""
    return torch.stack((v[..., 1], v[..., 2], v[..., 0]), dim=-1)


def spherical_harmonics_l(
    ell: int,
    vectors: Tensor,
    *,
    normalize: bool = True,
    normalization: str = "integral",
) -> Tensor:
    r"""Evaluate the real spherical harmonics of a single degree ``ell``.

    Parameters
    ----------
    ell:
        Degree, ``0 <= ell <= MAX_L``.
    vectors:
        Tensor of shape ``(..., 3)`` holding Cartesian vectors in the *standard*
        frame (polar axis :math:`z`).
    normalize:
        Project onto the unit sphere first.  The harmonics are only defined on
        :math:`S^2`; pass ``False`` only if ``vectors`` is already normalised.
    normalization:
        One of ``"integral"``, ``"component"``, ``"norm"`` -- see
        :func:`normalization_factor`.

    Returns
    -------
    Tensor of shape ``(..., 2 * ell + 1)`` ordered ``m = -ell .. +ell``.
    """
    if not 0 <= ell <= MAX_L:
        raise ValueError(f"closed forms are only written out for 0 <= l <= {MAX_L}, got {ell}")
    if vectors.shape[-1] != 3:
        raise ValueError(f"expected trailing dimension 3, got shape {tuple(vectors.shape)}")

    if normalize:
        # A plain `/ norm` would produce NaNs (and NaN gradients) at the origin.  Edge
        # vectors between distinct atoms are never zero, but self-loops and padding can
        # be, so clamp rather than trust the caller.
        norm = vectors.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(vectors.dtype).tiny)
        vectors = vectors / norm

    x, y, z = vectors.unbind(-1)
    scale = normalization_factor(ell, normalization)

    if ell == 0:
        out = [torch.full_like(x, 0.5 / math.sqrt(math.pi))]

    elif ell == 1:
        c = math.sqrt(3.0 / (4.0 * math.pi))
        # m = -1, 0, +1  ->  y, z, x
        out = [c * y, c * z, c * x]

    elif ell == 2:
        c2 = 0.5 * math.sqrt(15.0 / math.pi)
        c1 = 0.25 * math.sqrt(5.0 / math.pi)
        c0 = 0.25 * math.sqrt(15.0 / math.pi)
        out = [
            c2 * x * y,  # m = -2
            c2 * y * z,  # m = -1
            c1 * (3.0 * z * z - 1.0),  # m =  0   (3z^2 - r^2, r = 1)
            c2 * x * z,  # m = +1
            c0 * (x * x - y * y),  # m = +2
        ]

    else:  # ell == 3
        a = 0.25 * math.sqrt(35.0 / (2.0 * math.pi))
        b = 0.5 * math.sqrt(105.0 / math.pi)
        c = 0.25 * math.sqrt(21.0 / (2.0 * math.pi))
        d = 0.25 * math.sqrt(7.0 / math.pi)
        e = 0.25 * math.sqrt(105.0 / math.pi)
        out = [
            a * y * (3.0 * x * x - y * y),  # m = -3
            b * x * y * z,  # m = -2
            c * y * (5.0 * z * z - 1.0),  # m = -1
            d * z * (5.0 * z * z - 3.0),  # m =  0
            c * x * (5.0 * z * z - 1.0),  # m = +1
            e * z * (x * x - y * y),  # m = +2
            a * x * (x * x - 3.0 * y * y),  # m = +3
        ]

    stacked = torch.stack(out, dim=-1)
    return stacked * scale if scale != 1.0 else stacked


def spherical_harmonics(
    ells: int | list[int],
    vectors: Tensor,
    *,
    normalize: bool = True,
    normalization: str = "integral",
) -> Tensor:
    """Evaluate and concatenate several degrees at once.

    With ``ells=[0, 1, 2]`` the result has trailing dimension ``1 + 3 + 5 = 9``,
    laid out in increasing ``ell`` and, within each block, ``m = -ell .. +ell``.
    """
    if isinstance(ells, int):
        ells = list(range(ells + 1))
    parts = [
        spherical_harmonics_l(ell, vectors, normalize=normalize, normalization=normalization)
        for ell in ells
    ]
    return torch.cat(parts, dim=-1)
