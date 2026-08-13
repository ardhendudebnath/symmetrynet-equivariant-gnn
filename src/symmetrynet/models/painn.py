r"""PaiNN: equivariance from vector algebra instead of tensor products.

Schütt, Unke & Gastegger, *Equivariant message passing for the prediction of tensorial
properties and molecular spectra* (2021).

Why this is worth implementing alongside the TFN
------------------------------------------------
The Tensor Field Network in :mod:`symmetrynet.models.tfn` buys equivariance with
Clebsch-Gordan tensor products, which is the fully general construction: it works for any
:math:`\ell`, and the price is a per-edge contraction that dominates both runtime and
memory.

PaiNN observes that if you restrict yourself to :math:`\ell = 0` and :math:`\ell = 1` --
scalars and vectors -- you do not need the general machinery at all, because ordinary
vector algebra already supplies every equivariant operation you need:

* scaling a vector by an invariant scalar is equivariant;
* summing vectors is equivariant;
* the dot product of two vectors is invariant;
* the norm of a vector is invariant.

Every operation below is one of those four. There is no tensor product anywhere in this
file, no Wigner-3j table, and no ``e3nn`` import — and yet the network is exactly
equivariant, which the shared test suite checks against the same tolerance as the TFN.

The trade-off is real and worth naming: PaiNN cannot represent :math:`\ell \ge 2` features,
so the ablation showing that ``l_max=2`` beats ``l_max=1`` is evidence *against* this
design. It wins anyway on QM9 because it spends the saved compute on width and depth
instead — which is the interesting lesson.

Feature layout
--------------
Each atom carries scalars ``s`` of shape ``(N, F)`` and vectors ``v`` of shape
``(N, 3, F)``. Linear layers act on the trailing channel axis ``F`` only; the spatial axis
of length 3 is never mixed, which is precisely why they stay equivariant. Vectors are
initialised to zero: an isolated atom has no distinguished direction.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..nn.radial import BesselBasis, CosineCutoff
from ..utils.graph import radius_graph, scatter_sum

__all__ = ["PaiNN", "PaiNNMessage", "PaiNNUpdate"]

#: Guards the gradient of ``sqrt`` at zero, which is otherwise infinite.  Vector features
#: start at exactly zero, so this is hit on the very first forward pass, not rarely.
_NORM_EPS = 1e-8


def _safe_norm(vectors: Tensor, dim: int = 1) -> Tensor:
    """Euclidean norm over ``dim`` with a gradient that survives zero vectors."""
    return torch.sqrt(vectors.pow(2).sum(dim=dim) + _NORM_EPS)


class PaiNNMessage(nn.Module):
    r"""Continuous-filter message passing over scalars and vectors jointly.

    Splits a per-edge filter into three channel groups and uses them for three different
    jobs:

    * ``ss`` — updates the neighbour's scalars, exactly as in SchNet;
    * ``vv`` — rescales the neighbour's existing vectors (equivariant: scalar × vector);
    * ``vs`` — creates *new* vector features pointing along :math:`\hat r_{ij}`
      (equivariant: scalar × unit vector).

    The third is what lets directional information enter the network at all, and it is
    the direct counterpart of the TFN's spherical-harmonic edge embedding — just
    restricted to :math:`\ell = 1`, where the harmonic *is* the unit vector.
    """

    def __init__(self, hidden: int, num_radial: int, *, avg_num_neighbors: float = 12.0):
        super().__init__()
        self.hidden = hidden
        self.avg_num_neighbors = avg_num_neighbors
        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3 * hidden)
        )
        # No bias: the filter must vanish with the cutoff envelope, otherwise messages
        # would not decay smoothly to zero as an atom leaves the cutoff sphere.
        self.filter_net = nn.Linear(num_radial, 3 * hidden, bias=False)

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        edge_index: Tensor,
        radial: Tensor,
        envelope: Tensor,
        edge_dir: Tensor,
        num_nodes: int,
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]

        gates = self.scalar_mlp(s)[src] * (self.filter_net(radial) * envelope.unsqueeze(-1))
        gate_ss, gate_vv, gate_vs = torch.split(gates, self.hidden, dim=-1)

        # (E, F)
        message_s = gate_ss
        # (E, 3, F): rescale the neighbour's vectors, then add a new vector along the bond
        message_v = v[src] * gate_vv.unsqueeze(1) + edge_dir.unsqueeze(-1) * gate_vs.unsqueeze(1)

        norm = math.sqrt(self.avg_num_neighbors)
        s = s + scatter_sum(message_s, dst, num_nodes) / norm
        v = v + scatter_sum(message_v, dst, num_nodes) / norm
        return s, v


class PaiNNUpdate(nn.Module):
    r"""Node-wise mixing of scalars and vectors.

    Two learned channel mixtures ``U`` and ``V`` of the vector features feed three
    quantities:

    * :math:`\lVert Vv \rVert` — invariant, so it may be concatenated onto the scalars;
    * :math:`\langle Uv, Vv \rangle` — invariant, and the only route by which *directional*
      information reaches the scalar channels;
    * :math:`a_{vv} \odot Uv` — equivariant, the vector update.

    The dot-product term is the important one. Without it the vector features would evolve
    but never influence the prediction, and the model would collapse to a distance-only
    network with extra parameters.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.proj_u = nn.Linear(hidden, hidden, bias=False)
        self.proj_v = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3 * hidden)
        )

    def forward(self, s: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        u_v = self.proj_u(v)  # (N, 3, F)
        w_v = self.proj_v(v)  # (N, 3, F)

        stacked = torch.cat([s, _safe_norm(w_v)], dim=-1)  # (N, 2F) -- both invariant
        a_vv, a_sv, a_ss = torch.split(self.mlp(stacked), self.hidden, dim=-1)

        delta_v = a_vv.unsqueeze(1) * u_v
        delta_s = a_sv * (u_v * w_v).sum(dim=1) + a_ss
        return s + delta_s, v + delta_v


class PaiNN(nn.Module):
    """Polarizable Atom Interaction Neural Network for a single scalar target."""

    def __init__(
        self,
        *,
        num_species: int = 5,
        hidden: int = 128,
        num_layers: int = 3,
        cutoff: float = 5.0,
        num_radial: int = 20,
        avg_num_neighbors: float = 12.0,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.hidden = hidden

        self.embedding = nn.Embedding(num_species, hidden)
        self.radial_basis = BesselBasis(num_radial, cutoff)
        # PaiNN uses the cosine envelope; kept faithful to the paper rather than
        # switched to the polynomial one used elsewhere in this repo.
        self.envelope = CosineCutoff(cutoff)

        self.messages = nn.ModuleList(
            PaiNNMessage(hidden, num_radial, avg_num_neighbors=avg_num_neighbors)
            for _ in range(num_layers)
        )
        self.updates = nn.ModuleList(PaiNNUpdate(hidden) for _ in range(num_layers))

        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1)
        )

    def forward(
        self,
        species: Tensor,
        pos: Tensor,
        batch: Tensor | None = None,
        edge_index: Tensor | None = None,
    ) -> Tensor:
        num_nodes = pos.shape[0]
        if batch is None:
            batch = pos.new_zeros(num_nodes, dtype=torch.long)
        if edge_index is None:
            edge_index = radius_graph(pos, self.cutoff, batch)

        src, dst = edge_index[0], edge_index[1]
        edge_vec = pos[dst] - pos[src]
        edge_len = edge_vec.norm(dim=-1)
        # clamp_min keeps the direction finite if two atoms ever coincide.
        edge_dir = edge_vec / edge_len.clamp_min(_NORM_EPS).unsqueeze(-1)

        radial = self.radial_basis(edge_len)
        envelope = self.envelope(edge_len)

        s = self.embedding(species)
        v = pos.new_zeros(num_nodes, 3, self.hidden)

        for message, update in zip(self.messages, self.updates, strict=True):
            s, v = message(s, v, edge_index, radial, envelope, edge_dir, num_nodes)
            s, v = update(s, v)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        return scatter_sum(self.readout(s), batch, num_graphs).squeeze(-1)
