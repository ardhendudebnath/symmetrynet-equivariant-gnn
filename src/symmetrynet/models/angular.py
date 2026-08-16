r"""An **invariant** network that nevertheless sees bond angles.

The control this supplies
-------------------------
The other models in this project confound two things. Every equivariant model here also
has access to angular information, and the only angle-free model is also the only
non-equivariant one:

===========================  ==============  ============
model                        equivariant?    sees angles?
===========================  ==============  ============
``InvariantGNN`` (baseline)  invariant       no
``PaiNN`` / ``TFN``          equivariant     yes
===========================  ==============  ============

So when PaiNN beats the baseline, the cause is ambiguous: is it equivariance, or merely
angular information? This class fills the missing cell -- **invariant, but angle-aware** --
and separates them. If it matches PaiNN, equivariance per se contributed nothing and the
win was angular information all along. If PaiNN still leads, equivariance is doing
something beyond supplying angles.

Why Legendre polynomials are the right angular basis
----------------------------------------------------
Angles enter through :math:`P_{\ell}(\cos\theta_{jik})`, and that is not an arbitrary
choice. The spherical harmonic addition theorem states

.. math::
    \sum_{m} Y_{\ell m}(\hat r_1)\, Y_{\ell m}^{*}(\hat r_2)
    \;=\; \frac{2\ell+1}{4\pi}\, P_{\ell}(\cos\theta),

so :math:`P_\ell(\cos\theta)` is *exactly* the rotationally invariant contraction of two
degree-:math:`\ell` spherical harmonics -- the :math:`\ell \otimes \ell \to 0` path of a
Clebsch-Gordan tensor product, written without the machinery.

That makes this a genuinely fair control rather than a weakened one. It receives the same
angular content an equivariant network builds, in invariant form. What it *cannot* do is
carry that information forward as an :math:`\ell > 0` feature: angles are collapsed to
scalars at the moment they are read, so a later layer cannot compose them geometrically.
That difference -- not access to angles -- is what equivariance actually contributes, and
this model isolates it.

Structurally this is a Behler-Parrinello angular symmetry function in the message-passing
setting, and closely related to DimeNet's triplet messages.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..nn.radial import BesselBasis, PolynomialCutoff
from ..utils.graph import radius_graph, scatter_sum
from .baseline import InteractionBlock

__all__ = ["AngularInvariantGNN", "AngularBlock", "legendre_basis", "build_triplets"]


def legendre_basis(cos_theta: Tensor, degree: int) -> Tensor:
    r"""Legendre polynomials :math:`P_0 \dots P_{\ell}` evaluated at :math:`\cos\theta`.

    Built by the standard recurrence
    :math:`(\ell+1) P_{\ell+1} = (2\ell+1)\, x\, P_{\ell} - \ell\, P_{\ell-1}`, which is
    numerically stable and avoids writing out closed forms.

    Returns shape ``(..., degree + 1)``.
    """
    # Round-off in the dot product can push |cos| a hair past 1, where the recurrence is
    # still finite but the values stop being meaningful.
    x = cos_theta.clamp(-1.0, 1.0)
    polys = [torch.ones_like(x)]
    if degree >= 1:
        polys.append(x)
    for order in range(1, degree):
        polys.append(
            ((2 * order + 1) * x * polys[order] - order * polys[order - 1]) / (order + 1)
        )
    return torch.stack(polys, dim=-1)


def build_triplets(edge_index: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]:
    """Index pairs of edges that share a destination atom.

    For every atom ``i`` this enumerates ordered pairs of incoming edges ``(j -> i, k -> i)``
    with ``j != k``, which is what an angle at ``i`` is defined over.

    Returns ``(edge_a, edge_b)``, indices into ``edge_index``, both sharing a destination.
    """
    dst = edge_index[1]
    num_edges = dst.numel()
    device = dst.device
    if num_edges == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    # Group edges by destination so each atom's incoming edges are contiguous.
    order = torch.argsort(dst)
    dst_sorted = dst[order]
    counts = torch.bincount(dst_sorted, minlength=num_nodes)
    starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])

    # Each edge pairs with every edge sharing its destination.
    per_edge = counts[dst_sorted]
    edge_a = torch.arange(num_edges, device=device).repeat_interleave(per_edge)

    # Position within the destination's block, so we can index its siblings.
    block_starts = torch.cat([per_edge.new_zeros(1), per_edge.cumsum(0)[:-1]])
    local = torch.arange(int(per_edge.sum()), device=device) - block_starts.repeat_interleave(
        per_edge
    )
    edge_b = starts[dst_sorted[edge_a]] + local

    keep = edge_a != edge_b  # an atom forms no angle with itself
    return order[edge_a[keep]], order[edge_b[keep]]


class AngularBlock(nn.Module):
    r"""Aggregates triplet messages weighted by :math:`P_\ell(\cos\theta)`.

    The angular width is kept well below the scalar width on purpose: the number of
    triplets grows as the square of the degree (~430k for a batch of 96 QM9 molecules at a
    5 A cutoff), so this is the memory-critical part of the model.
    """

    def __init__(
        self,
        hidden: int,
        num_radial: int,
        *,
        angular_hidden: int = 32,
        max_degree: int = 4,
        avg_num_triplets: float = 200.0,
    ):
        super().__init__()
        self.angular_hidden = angular_hidden
        self.max_degree = max_degree
        self.avg_num_triplets = avg_num_triplets

        self.project_in = nn.Linear(hidden, angular_hidden, bias=False)
        # One filter per (radial, angular) pair, mirroring how the tensor product mixes a
        # distance-dependent weight with an angular basis.
        self.filter_net = nn.Sequential(
            nn.Linear(2 * num_radial + (max_degree + 1), 64),
            nn.SiLU(),
            nn.Linear(64, angular_hidden),
        )
        self.project_out = nn.Sequential(
            nn.Linear(angular_hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        radial: Tensor,
        envelope: Tensor,
        edge_vec: Tensor,
        edge_len: Tensor,
        num_nodes: int,
    ) -> Tensor:
        edge_a, edge_b = build_triplets(edge_index, num_nodes)
        if edge_a.numel() == 0:
            return x

        # Angle at the shared destination atom, between the two bonds meeting there.
        va, vb = edge_vec[edge_a], edge_vec[edge_b]
        denom = (edge_len[edge_a] * edge_len[edge_b]).clamp_min(1e-9)
        cos_theta = (va * vb).sum(-1) / denom
        angular = legendre_basis(cos_theta, self.max_degree)

        features = torch.cat([radial[edge_a], radial[edge_b], angular], dim=-1)
        weights = self.filter_net(features) * (envelope[edge_a] * envelope[edge_b]).unsqueeze(-1)

        # Both source atoms contribute; their product is what makes this a genuine
        # three-body term rather than two independent two-body ones.
        src = edge_index[0]
        projected = self.project_in(x)
        messages = projected[src[edge_a]] * projected[src[edge_b]] * weights

        target = edge_index[1][edge_a]
        pooled = scatter_sum(messages, target, num_nodes) / math.sqrt(self.avg_num_triplets)
        return x + self.project_out(pooled)


class AngularInvariantGNN(nn.Module):
    """Distance *and* angle aware, and invariant throughout.

    Deliberately the same skeleton as :class:`~symmetrynet.models.baseline.InvariantGNN`:
    identical embedding, radial basis, cutoff, interaction blocks and readout, with angular
    blocks interleaved. The only difference between the two models is whether angles are
    visible, which is what makes the pair a clean control.
    """

    def __init__(
        self,
        *,
        num_species: int = 5,
        hidden: int = 128,
        num_layers: int = 4,
        cutoff: float = 5.0,
        num_radial: int = 8,
        radial_hidden: int = 128,
        angular_hidden: int = 32,
        max_degree: int = 4,
        avg_num_neighbors: float = 12.0,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.embedding = nn.Embedding(num_species, hidden)
        self.radial_basis = BesselBasis(num_radial, cutoff)
        self.envelope = PolynomialCutoff(cutoff)

        # Triplets per atom scale as the square of the neighbour count.
        avg_triplets = max(1.0, avg_num_neighbors * (avg_num_neighbors - 1))

        self.interactions = nn.ModuleList(
            InteractionBlock(hidden, num_radial, radial_hidden=radial_hidden,
                             avg_num_neighbors=avg_num_neighbors)
            for _ in range(num_layers)
        )
        self.angular = nn.ModuleList(
            AngularBlock(hidden, num_radial, angular_hidden=angular_hidden,
                         max_degree=max_degree, avg_num_triplets=avg_triplets)
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
        edge_vec = pos[dst] - pos[src]
        edge_len = edge_vec.norm(dim=-1)
        radial = self.radial_basis(edge_len)
        envelope = self.envelope(edge_len)

        x = self.embedding(species)
        for interaction, angular in zip(self.interactions, self.angular, strict=True):
            x = interaction(x, edge_index, radial, envelope, num_nodes)
            x = angular(x, edge_index, radial, envelope, edge_vec, edge_len, num_nodes)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        return scatter_sum(self.readout(x), batch, num_graphs).squeeze(-1)
