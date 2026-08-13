"""Precision helpers for equivariance testing.

Background: a hard-won detail
-----------------------------
``e3nn`` bakes Wigner-3j coefficients into each ``TensorProduct`` when the module is
**constructed**, at whatever ``torch`` default dtype is active at that moment.  Calling
``model.double()`` afterwards converts parameters and buffers but *not* those baked
constants -- it merely widens already-truncated float32 values.

The consequence is a trap.  A model built at float32 and cast to float64 shows an
equivariance residual of roughly:

===========  ==========================  =====================
``l_max``    cast after construction     built in float64
===========  ==========================  =====================
1            1.3e-15                     2.7e-15
2            2.1e-09                     1.3e-15
3            8.0e-09                     1.3e-15
===========  ==========================  =====================

Six orders of magnitude, appearing only for ``l_max >= 2`` -- easily misread as "the
higher-degree paths are subtly wrong" when in fact the mathematics is exact and only the
constants were rounded.  :func:`default_dtype` makes the correct construction order easy.

None of this affects *training*, which runs in float32 where 1e-9 is far below the
rounding floor anyway.  It matters only when verifying equivariance, where the whole
point is to distinguish "exact up to rounding" from "nearly right".
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

__all__ = ["default_dtype"]


@contextmanager
def default_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Temporarily set the global default dtype.

    Construct ``e3nn``-based modules inside this block when they must be numerically
    exact::

        with default_dtype(torch.float64):
            model = TensorFieldNetwork(l_max=2)
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)
