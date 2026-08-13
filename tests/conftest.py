"""Shared fixtures.

Equivariance tests run in ``float64``.  In ``float32`` the residual is ~1e-6 and it
becomes impossible to distinguish "exactly equivariant, limited by rounding" from
"almost equivariant because of a subtle bug".  Double precision makes the distinction
sharp: a correct model lands at ~1e-15, a broken one does not.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(20260812)


@pytest.fixture
def molecules():
    """A small batch of three synthetic 'molecules' with realistic bond lengths."""
    sizes = [12, 9, 15]
    species = torch.cat([torch.randint(0, 5, (n,)) for n in sizes])
    pos = torch.cat([torch.randn(n, 3, dtype=torch.float64) * 1.6 for n in sizes])
    batch = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(sizes)])
    return species, pos, batch, sizes
