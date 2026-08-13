r"""Clebsch-Gordan coefficients for SO(3), in the **real** spherical-harmonic basis.

This is the mathematical heart of the project, and it is built from first principles
rather than imported from ``e3nn``.

What a Clebsch-Gordan tensor actually is
----------------------------------------
Take a feature :math:`u` living in irrep :math:`\ell_1` (dimension :math:`2\ell_1+1`)
and a feature :math:`v` living in irrep :math:`\ell_2`.  Their outer product
:math:`u \otimes v` lives in a :math:`(2\ell_1+1)(2\ell_2+1)`-dimensional space, which
is *not* irreducible -- it decomposes as

.. math::  \ell_1 \otimes \ell_2 \;=\; \bigoplus_{\ell=|\ell_1-\ell_2|}^{\ell_1+\ell_2} \ell .

The Clebsch-Gordan tensor :math:`C^{(\ell_1 \ell_2 \ell)}_{m_1 m_2 m}` is exactly
the projector onto the :math:`\ell` summand.  Contracting

.. math::  w_m = \sum_{m_1 m_2} C_{m_1 m_2 m}\, u_{m_1} v_{m_2}

produces a :math:`w` that transforms in irrep :math:`\ell`.  Equivalently, as a matrix
:math:`C` of shape :math:`((2\ell_1+1)(2\ell_2+1), 2\ell+1)` it *intertwines* the two
representations:

.. math::  \bigl(D^{\ell_1}(R) \otimes D^{\ell_2}(R)\bigr)\, C \;=\; C\, D^{\ell}(R)
           \quad \text{for every rotation } R .

That identity is the entire reason equivariant networks work, and
``tests/test_clebsch_gordan.py`` checks it numerically rather than taking it on faith.

How this module computes them
-----------------------------
1. Complex CG coefficients :math:`\langle \ell_1 m_1 \ell_2 m_2 | \ell m \rangle` come
   from the **Racah formula** -- a closed-form alternating factorial sum.
2. The real harmonics are related to the complex ones by a fixed unitary
   :math:`U^{\ell}` (:func:`real_basis_change`), so conjugating the complex tensor by
   :math:`U` gives the real-basis coefficients:

   .. math::
       C_{\mathrm{real}} = (U^{\ell_1} \otimes U^{\ell_2})\,
                           C_{\mathrm{cplx}}\, (U^{\ell})^{\dagger}.

   The result is real up to a single global phase, which we strip off; a global phase
   is physically meaningless here because it is absorbed by the learned weights.

Everything is cached, computed in ``float64``/``complex128``, and only downcast at the
point of use.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "clebsch_gordan_element",
    "clebsch_gordan",
    "real_basis_change",
    "wigner_3j",
    "decomposition_degrees",
]


def _fact(n: int) -> float:
    return float(math.factorial(n))


def decomposition_degrees(l1: int, l2: int, l_max: int | None = None) -> list[int]:
    """Degrees appearing in ``l1 ⊗ l2``, optionally truncated at ``l_max``.

    This is the *selection rule* ``|l1 - l2| <= l <= l1 + l2`` -- the reason a network
    with only ``l=0`` features can never generate angular information, and the reason
    ``l_max`` controls how much geometry the model can represent.
    """
    hi = l1 + l2 if l_max is None else min(l1 + l2, l_max)
    return list(range(abs(l1 - l2), hi + 1))


def clebsch_gordan_element(l1: int, m1: int, l2: int, m2: int, l3: int, m3: int) -> float:
    r"""One complex coefficient :math:`\langle l_1 m_1 l_2 m_2 | l_3 m_3 \rangle`.

    Implements the Racah closed form.  Integer degrees only -- SO(3) irreps of molecules
    are integer-:math:`\ell`; half-integer spin never enters this project.
    """
    if m3 != m1 + m2:
        return 0.0
    if not abs(l1 - l2) <= l3 <= l1 + l2:
        return 0.0
    if abs(m1) > l1 or abs(m2) > l2 or abs(m3) > l3:
        return 0.0

    prefactor = math.sqrt(
        (2 * l3 + 1)
        * _fact(l3 + l1 - l2)
        * _fact(l3 - l1 + l2)
        * _fact(l1 + l2 - l3)
        / _fact(l1 + l2 + l3 + 1)
    ) * math.sqrt(
        _fact(l3 + m3)
        * _fact(l3 - m3)
        * _fact(l1 - m1)
        * _fact(l1 + m1)
        * _fact(l2 - m2)
        * _fact(l2 + m2)
    )

    # The sum runs over the k for which every factorial argument is non-negative.
    k_lo = max(0, l2 - l3 - m1, l1 - l3 + m2)
    k_hi = min(l1 + l2 - l3, l1 - m1, l2 + m2)

    total = 0.0
    for k in range(k_lo, k_hi + 1):
        denom = (
            _fact(k)
            * _fact(l1 + l2 - l3 - k)
            * _fact(l1 - m1 - k)
            * _fact(l2 + m2 - k)
            * _fact(l3 - l2 + m1 + k)
            * _fact(l3 - l1 - m2 + k)
        )
        total += (-1.0) ** k / denom

    return prefactor * total


@cache
def _complex_cg(l1: int, l2: int, l3: int) -> np.ndarray:
    """Complex-basis CG tensor, shape ``(2l1+1, 2l2+1, 2l3+1)``, indices ordered ``m=-l..l``."""
    out = np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1), dtype=np.float64)
    for i1, m1 in enumerate(range(-l1, l1 + 1)):
        for i2, m2 in enumerate(range(-l2, l2 + 1)):
            m3 = m1 + m2
            if abs(m3) <= l3:
                out[i1, i2, m3 + l3] = clebsch_gordan_element(l1, m1, l2, m2, l3, m3)
    return out


@cache
def _real_basis_change_np(ell: int) -> np.ndarray:
    r"""Unitary :math:`U^{\ell}` with :math:`Y^{\mathrm{real}} = U\, Y^{\mathrm{cplx}}`.

    The standard relation between real and complex spherical harmonics:

    .. math::
        Y_{\ell m}^{\mathrm{real}} =
        \begin{cases}
          \tfrac{i}{\sqrt 2}\bigl(Y_\ell^{m} - (-1)^m Y_\ell^{-m}\bigr) & m < 0 \\
          Y_\ell^{0} & m = 0 \\
          \tfrac{1}{\sqrt 2}\bigl(Y_\ell^{-m} + (-1)^m Y_\ell^{m}\bigr) & m > 0
        \end{cases}
    """
    dim = 2 * ell + 1
    u = np.zeros((dim, dim), dtype=np.complex128)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for m in range(-ell, ell + 1):
        row = m + ell
        if m == 0:
            u[row, ell] = 1.0
        elif m > 0:
            u[row, -m + ell] = inv_sqrt2
            u[row, m + ell] = ((-1.0) ** m) * inv_sqrt2
        else:  # m < 0
            am = -m
            u[row, -am + ell] = 1j * inv_sqrt2
            u[row, am + ell] = -1j * ((-1.0) ** am) * inv_sqrt2
    return u


def _strip_global_phase(arr: np.ndarray, *, tol: float = 1e-9) -> np.ndarray:
    """Return a real array equal to ``arr`` up to an overall phase of 1 or i.

    The real-basis CG tensor comes out either purely real or purely imaginary depending
    on the parity of ``l1 + l2 + l3``.  An overall constant multiplying an intertwiner
    leaves the intertwining property intact, so discarding the phase is legitimate --
    and it keeps every downstream tensor in real arithmetic.
    """
    max_re = float(np.abs(arr.real).max())
    max_im = float(np.abs(arr.imag).max())
    scale = max(max_re, max_im)
    if scale < tol:
        return np.zeros(arr.shape, dtype=np.float64)
    if max_im <= tol * scale:
        return arr.real.astype(np.float64)
    if max_re <= tol * scale:
        return arr.imag.astype(np.float64)
    raise RuntimeError(
        "real-basis CG tensor is genuinely complex "
        f"(max|Re|={max_re:.3e}, max|Im|={max_im:.3e}); the basis change is wrong"
    )


@cache
def _real_cg(l1: int, l2: int, l3: int) -> np.ndarray:
    """Real-basis CG tensor, shape ``(2l1+1, 2l2+1, 2l3+1)``."""
    if not abs(l1 - l2) <= l3 <= l1 + l2:
        raise ValueError(
            f"l={l3} is outside the selection rule |{l1}-{l2}| <= l <= {l1}+{l2}; "
            "this path carries no information"
        )
    u1 = _real_basis_change_np(l1)
    u2 = _real_basis_change_np(l2)
    u3 = _real_basis_change_np(l3)
    cplx = _complex_cg(l1, l2, l3).astype(np.complex128)
    # C_real = (U1 (x) U2) C_cplx (U3)^dagger
    transformed = np.einsum("ap,bq,pqr,cr->abc", u1, u2, cplx, u3.conj())
    real = _strip_global_phase(transformed)

    # Fix the residual sign freedom deterministically so repeated runs agree.
    flat = real.reshape(-1)
    nz = np.flatnonzero(np.abs(flat) > 1e-12)
    if nz.size and flat[nz[0]] < 0:
        real = -real
    return np.ascontiguousarray(real)


def real_basis_change(ell: int, *, device: torch.device | str | None = None) -> Tensor:
    """The complex-to-real change of basis :math:`U^{\\ell}` as a complex tensor."""
    return torch.from_numpy(_real_basis_change_np(ell).copy()).to(device=device)


def clebsch_gordan(
    l1: int,
    l2: int,
    l3: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    r"""Real-basis CG tensor of shape ``(2l1+1, 2l2+1, 2l3+1)``.

    Normalised so the matrix of shape ``((2l1+1)(2l2+1), 2l3+1)`` has **orthonormal
    columns**, i.e. :math:`C^{\top} C = I`.  This is the convention in which
    :math:`C^{\top} (D^{\ell_1} \otimes D^{\ell_2}) C = D^{\ell_3}` holds exactly, which is
    what :mod:`symmetrynet.scratch.wigner` relies on.
    """
    return torch.from_numpy(_real_cg(l1, l2, l3).copy()).to(dtype=dtype, device=device)


def wigner_3j(
    l1: int,
    l2: int,
    l3: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    r"""The same tensor rescaled to unit Frobenius norm (``e3nn``'s convention).

    ``e3nn.o3.wigner_3j`` normalises so that :math:`\sum_{m_1 m_2 m_3} w^2 = 1`, whereas
    the orthonormal-column CG convention gives :math:`\sum C^2 = 2\ell_3 + 1`.  The two
    therefore differ by :math:`\sqrt{2\ell_3+1}` (and possibly an overall sign, which is
    pure convention).
    """
    cg = clebsch_gordan(l1, l2, l3, dtype=dtype, device=device)
    return cg / math.sqrt(2 * l3 + 1)
