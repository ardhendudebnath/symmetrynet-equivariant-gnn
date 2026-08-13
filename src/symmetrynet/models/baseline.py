r"""Phase 1 baseline: a distance-only invariant message-passing network (SchNet-style).

This is the **control condition** for the whole project, so it is written to be a fair
opponent rather than a strawman.  It shares the radial basis, cutoff envelope,
aggregation normalisation, readout structure and training loop with the equivariant
model.  The *only* difference is what an edge is allowed to know:

* here, an edge carries :math:`\lVert r_{ij} \rVert` -- a single invariant number;
* in the TFN, it carries :math:`Y_{\ell}(\hat r_{ij})` -- the full direction.

That makes the comparison interpretable.  Any accuracy gap is attributable to angular
information, not to a better-tuned optimiser or a larger radial network.

What this model provably cannot represent
-----------------------------------------
Because every input is a distance, the network is invariant to rotations *by
construction* -- which is good -- but it is also blind to anything distances alone do
not determine.  A single node's set of neighbour distances does not fix the bond
*angles*, so two genuinely different local geometries can produce byte-identical inputs.
Deeper message passing recovers some of this indirectly, which is why the baseline is
competitive rather than hopeless, but the limitation is real and is exactly what the
equivariant model is meant to fix.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..nn.radial import BesselBasis, PolynomialCutoff
from ..utils.graph import radius_graph, scatter_sum

__all__ = ["InvariantGNN", "InteractionBlock"]


class InteractionBlock(nn.Module):
    """SchNet continuous-filter convolution: a distance-dependent elementwise filter."""

    def __init__(
        self,
        hidden: int,
        num_radial: int,
        *,
        radial_hidden: int = 128,
        avg_num_neighbors: float = 12.0,
    ):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.filter_net = nn.Sequential(
            nn.Linear(num_radial, radial_hidden),
            nn.SiLU(),
            nn.Linear(radial_hidden, hidden),
        )
        self.lin_in = nn.Linear(hidden, hidden, bias=False)
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        radial_features: Tensor,
        envelope: Tensor,
        num_nodes: int,
    ) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        filters = self.filter_net(radial_features) * envelope.unsqueeze(-1)
        messages = self.lin_in(x)[src] * filters
        aggregated = scatter_sum(messages, dst, num_nodes) / math.sqrt(self.avg_num_neighbors)
        return x + self.mlp_out(aggregated)  # residual, as in SchNet


class InvariantGNN(nn.Module):
    """Distance-only invariant GNN over a radius graph."""

    def __init__(
        self,
        *,
        num_species: int = 5,
        hidden: int = 128,
        num_layers: int = 4,
        cutoff: float = 5.0,
        num_radial: int = 8,
        radial_hidden: int = 128,
        avg_num_neighbors: float = 12.0,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.embedding = nn.Embedding(num_species, hidden)
        self.radial_basis = BesselBasis(num_radial, cutoff)
        self.envelope = PolynomialCutoff(cutoff)
        self.interactions = nn.ModuleList(
            InteractionBlock(
                hidden,
                num_radial,
                radial_hidden=radial_hidden,
                avg_num_neighbors=avg_num_neighbors,
            )
            for _ in range(num_layers)
        )
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
        edge_len = (pos[dst] - pos[src]).norm(dim=-1)
        radial_features = self.radial_basis(edge_len)
        envelope = self.envelope(edge_len)

        x = self.embedding(species)
        for interaction in self.interactions:
            x = interaction(x, edge_index, radial_features, envelope, num_nodes)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        return scatter_sum(self.readout(x), batch, num_graphs).squeeze(-1)
