r"""The negative control: a model that sees raw coordinates and is *not* equivariant.

Every claim in this project is comparative, so it needs something that visibly fails.
This model is architecturally identical to :class:`~symmetrynet.models.baseline.InvariantGNN`
except that each edge carries the **raw components** of :math:`r_{ij}` alongside the
distance.  Those three numbers change when the molecule is rotated, so the prediction
changes too.

Two things make this the right control rather than a strawman:

* It has *more* information than the invariant baseline, not less, and slightly more
  capacity.  So when it fails the equivariance test, the failure is attributable to
  the missing symmetry constraint and not to a weaker model.
* It is exactly the thing a well-meaning practitioner writes when they "just feed in
  the coordinates".  Its rotation curve -- a wildly swinging prediction where the
  equivariant model draws a flat line -- is the demo plot the README leads with.

Note that ``NaiveCoordinateGNN`` remains permutation invariant and (optionally)
translation invariant.  Only rotational symmetry is broken, which isolates the variable
under study.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..nn.radial import BesselBasis, PolynomialCutoff
from ..utils.graph import radius_graph, scatter_mean, scatter_sum

__all__ = ["NaiveCoordinateGNN", "NaiveCoordinateMLP"]


class NaiveCoordinateGNN(nn.Module):
    """Message passing on raw relative-position components.  Rotation-variant by design."""

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
        center_positions: bool = True,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.avg_num_neighbors = avg_num_neighbors
        # Centring keeps the model translation invariant so that the rotation test
        # isolates rotational symmetry alone.
        self.center_positions = center_positions

        self.embedding = nn.Embedding(num_species, hidden)
        self.radial_basis = BesselBasis(num_radial, cutoff)
        self.envelope = PolynomialCutoff(cutoff)

        self.filter_nets = nn.ModuleList(
            nn.Sequential(
                nn.Linear(num_radial + 3, radial_hidden),  # +3: the raw vector components
                nn.SiLU(),
                nn.Linear(radial_hidden, hidden),
            )
            for _ in range(num_layers)
        )
        self.lin_in = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in range(num_layers)
        )
        self.mlp_out = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
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
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0

        if self.center_positions:
            pos = pos - scatter_mean(pos, batch, num_graphs)[batch]
        if edge_index is None:
            edge_index = radius_graph(pos, self.cutoff, batch)

        src, dst = edge_index[0], edge_index[1]
        edge_vec = pos[dst] - pos[src]
        edge_len = edge_vec.norm(dim=-1)
        envelope = self.envelope(edge_len)
        # The offending line: raw components, which rotate with the molecule.
        edge_feat = torch.cat([self.radial_basis(edge_len), edge_vec], dim=-1)

        x = self.embedding(species)
        for filter_net, lin, mlp in zip(self.filter_nets, self.lin_in, self.mlp_out, strict=True):
            filters = filter_net(edge_feat) * envelope.unsqueeze(-1)
            messages = lin(x)[src] * filters
            aggregated = scatter_sum(messages, dst, num_nodes) / math.sqrt(self.avg_num_neighbors)
            x = x + mlp(aggregated)

        return scatter_sum(self.readout(x), batch, num_graphs).squeeze(-1)


class NaiveCoordinateMLP(nn.Module):
    """An even blunter control: a DeepSets MLP over ``(one-hot species, x, y, z)``.

    Permutation invariant via the sum pooling, but nothing else.  Useful for showing
    just how badly an unconstrained model behaves under rotation.
    """

    def __init__(self, *, num_species: int = 5, hidden: int = 256, center_positions: bool = True):
        super().__init__()
        self.num_species = num_species
        self.center_positions = center_positions
        self.encoder = nn.Sequential(
            nn.Linear(num_species + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(
        self,
        species: Tensor,
        pos: Tensor,
        batch: Tensor | None = None,
        edge_index: Tensor | None = None,  # noqa: ARG002 - signature parity with the GNNs
    ) -> Tensor:
        if batch is None:
            batch = pos.new_zeros(pos.shape[0], dtype=torch.long)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        if self.center_positions:
            pos = pos - scatter_mean(pos, batch, num_graphs)[batch]

        one_hot = torch.nn.functional.one_hot(species, self.num_species).to(pos.dtype)
        node_feat = self.encoder(torch.cat([one_hot, pos], dim=-1))
        pooled = scatter_sum(node_feat, batch, num_graphs)
        return self.readout(pooled).squeeze(-1)
