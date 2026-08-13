r"""Radial basis functions and smooth cutoffs.

These act on the interatomic **distance**, which is already an invariant scalar, so
nothing here can break equivariance -- these are the same components a plain invariant
GNN uses.  Sharing them between the baseline and the equivariant model is deliberate:
it means any accuracy difference between the two comes from the angular machinery, not
from a better-tuned radial featurisation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["GaussianSmearing", "BesselBasis", "PolynomialCutoff", "CosineCutoff"]


class GaussianSmearing(nn.Module):
    r"""SchNet-style expansion :math:`\exp(-\gamma (d - \mu_k)^2)` on a fixed grid."""

    def __init__(self, num_basis: int = 50, cutoff: float = 5.0, trainable: bool = False):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_basis)
        spacing = float(centers[1] - centers[0])
        self.gamma = 0.5 / spacing**2
        if trainable:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers)
        self.num_basis = num_basis

    def forward(self, dist: Tensor) -> Tensor:
        diff = dist.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff.pow(2))


class BesselBasis(nn.Module):
    r"""Radial Bessel basis from DimeNet: :math:`\sqrt{2/c}\,\sin(n \pi d / c)/d`.

    Fewer functions than Gaussian smearing for the same accuracy, and it forms a proper
    orthogonal basis on :math:`[0, c]` rather than an arbitrary set of bumps.

    On ``normalize``
    ----------------
    The raw basis has values of order 0.3 over the physically occupied range of bond
    lengths, not order 1.  That is harmless on its own, but it propagates: the radial
    MLP that consumes it is initialised assuming unit-variance inputs, so its outputs
    come out ~8x too small, and those outputs are the *weights* of the equivariant
    tensor product.  ``e3nn``'s tensor product normalises on the assumption that its
    weights have unit variance, so the entire message pathway ends up attenuated,
    increasingly so with depth (measured: message/skip magnitude ratio falling 0.27 ->
    0.115 across four layers).  The symptom is a model that is exactly equivariant,
    trains stably, and simply underfits.

    Rescaling the basis to unit RMS over the cutoff sphere fixes the conditioning at the
    source.  It is applied to *both* models, since they share this module -- the
    comparison stays controlled.
    """

    def __init__(
        self,
        num_basis: int = 8,
        cutoff: float = 5.0,
        trainable: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.num_basis = num_basis
        freqs = torch.arange(1, num_basis + 1, dtype=torch.get_default_dtype()) * math.pi
        if trainable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)
        self.prefactor = math.sqrt(2.0 / cutoff)

        # Weight by r^2 dr: pair separations are distributed over a spherical shell, so
        # large distances are far more common than a uniform average would suggest.
        scale = 1.0
        if normalize:
            with torch.no_grad():
                grid = torch.linspace(1e-3, cutoff, 2048)
                raw = self.prefactor * torch.sin(freqs * grid.unsqueeze(-1) / cutoff) / (
                    grid.unsqueeze(-1)
                )
                weight = grid.pow(2)
                weight = weight / weight.sum()
                rms = (raw.pow(2).mean(-1) * weight).sum().sqrt()
                scale = float(1.0 / rms.clamp_min(1e-12))
        self.scale = scale

    def forward(self, dist: Tensor) -> Tensor:
        # clamp_min avoids a 0/0 at coincident atoms; real edges never hit it.
        d = dist.unsqueeze(-1).clamp_min(1e-8)
        return self.scale * self.prefactor * torch.sin(self.freqs * d / self.cutoff) / d


class PolynomialCutoff(nn.Module):
    r"""DimeNet's smooth envelope: :math:`1` at :math:`d=0`, and :math:`C^{2}`-zero at the cutoff.

    A hard distance cutoff would make the energy *discontinuous* as an atom crosses the
    boundary, which is unphysical and shows up as noise in the loss.  This envelope and
    its first two derivatives all vanish at :math:`d = c`.
    """

    def __init__(self, cutoff: float = 5.0, p: int = 6):
        super().__init__()
        self.cutoff = cutoff
        self.p = p

    def forward(self, dist: Tensor) -> Tensor:
        d = dist / self.cutoff
        p = self.p
        env = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * d.pow(p)
            + p * (p + 2.0) * d.pow(p + 1)
            - (p * (p + 1.0) / 2.0) * d.pow(p + 2)
        )
        return torch.where(dist < self.cutoff, env, torch.zeros_like(env))


class CosineCutoff(nn.Module):
    r"""Behler's cosine envelope :math:`\tfrac12(\cos(\pi d / c) + 1)`."""

    def __init__(self, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: Tensor) -> Tensor:
        env = 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0)
        return torch.where(dist < self.cutoff, env, torch.zeros_like(env))
