r"""A complete E(3)-equivariant message-passing layer and a small end-to-end model.

This is the Phase 2 deliverable: a Tensor Field Network convolution assembled entirely
from the hand-derived primitives in this subpackage, with no ``e3nn`` anywhere in the
call graph.

Why the layer is equivariant, in one paragraph
----------------------------------------------
Translations are handled by only ever looking at *relative* positions
:math:`r_{ij} = x_j - x_i`, which are unchanged by a global shift.  Rotations are handled
degree by degree: the edge direction enters only through
:math:`Y_{\ell}(\hat r_{ij})`, which rotates by :math:`D^{\ell}`; the edge *length* enters
only through a radial MLP, and a length is invariant; the two are combined by the
Clebsch-Gordan tensor, which intertwines by construction; the sum over neighbours is a
sum of equivariant terms and so is equivariant; and the nonlinearity only scales
higher-:math:`\ell` features by invariants.  Composing equivariant maps gives an
equivariant map, so the whole network is equivariant.  Reading off the
:math:`\ell = 0` channels at the end turns that into exact *invariance* of the
prediction -- not approximate, not learned, but true for every set of weights, including
random ones.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..nn.radial import BesselBasis, PolynomialCutoff
from ..utils.graph import radius_graph, scatter_sum
from .spherical_harmonics import spherical_harmonics_l
from .tensor_product import EquivariantLinear, Features, Gate, Irreps, TensorProduct

__all__ = ["EquivariantConv", "ScratchTFN"]


class EquivariantConv(nn.Module):
    """One TFN convolution: gather, tensor-product with the edge geometry, aggregate."""

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_sh: Irreps,
        irreps_out: Irreps,
        *,
        num_radial: int,
        radial_hidden: int = 64,
        avg_num_neighbors: float = 12.0,
        self_interaction: bool = True,
    ):
        super().__init__()
        self.tp = TensorProduct(irreps_in, irreps_sh, irreps_out)
        self.avg_num_neighbors = avg_num_neighbors

        self.radial = nn.Sequential(
            nn.Linear(num_radial, radial_hidden),
            nn.SiLU(),
            nn.Linear(radial_hidden, radial_hidden),
            nn.SiLU(),
            nn.Linear(radial_hidden, self.tp.weight_numel),
        )
        # Start near zero so the first forward pass is dominated by the (well-scaled)
        # self-interaction rather than by random tensor-product noise.
        nn.init.zeros_(self.radial[-1].bias)
        nn.init.normal_(self.radial[-1].weight, std=1.0 / math.sqrt(radial_hidden))

        shared = {ell: mul for ell, mul in irreps_out.items() if ell in irreps_in}
        self.self_interaction = (
            EquivariantLinear(irreps_in, shared) if self_interaction and shared else None
        )

    def forward(
        self,
        x: Features,
        sh: Features,
        edge_index: Tensor,
        radial_features: Tensor,
        envelope: Tensor,
        num_nodes: int,
    ) -> Features:
        src, dst = edge_index[0], edge_index[1]

        # Multiplying the radial weights by the envelope makes every message decay
        # smoothly to zero at the cutoff, so the model is continuous in the positions.
        weights = self.radial(radial_features) * envelope.unsqueeze(-1)

        x_src: Features = {ell: feat[src] for ell, feat in x.items()}
        messages = self.tp(x_src, sh, weights)

        out: Features = {
            ell: scatter_sum(msg, dst, num_nodes) / math.sqrt(self.avg_num_neighbors)
            for ell, msg in messages.items()
        }

        if self.self_interaction is not None:
            for ell, feat in self.self_interaction(x).items():
                out[ell] = out[ell] + feat
        return out


class ScratchTFN(nn.Module):
    """A compact Tensor Field Network built only from hand-derived primitives.

    Deliberately small.  Its job is to demonstrate -- and let the test suite verify --
    that the from-scratch machinery composes into a working, exactly equivariant model.
    The competitive model lives in :mod:`symmetrynet.models.tfn` and uses ``e3nn``.
    """

    def __init__(
        self,
        *,
        num_species: int = 5,
        hidden_multiplicity: int = 16,
        l_max: int = 2,
        num_layers: int = 2,
        cutoff: float = 5.0,
        num_radial: int = 8,
        radial_hidden: int = 64,
        avg_num_neighbors: float = 12.0,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.l_max = l_max
        self.sh_degrees = list(range(l_max + 1))

        self.embedding = nn.Embedding(num_species, hidden_multiplicity)
        self.radial_basis = BesselBasis(num_radial, cutoff)
        self.envelope = PolynomialCutoff(cutoff)

        irreps_sh: Irreps = {ell: 1 for ell in self.sh_degrees}
        irreps_hidden: Irreps = {ell: hidden_multiplicity for ell in self.sh_degrees}

        self.convs = nn.ModuleList()
        self.gates = nn.ModuleList()
        irreps_in: Irreps = {0: hidden_multiplicity}
        for layer in range(num_layers):
            # The readout only ever reads l=0, so the final convolution emits scalars
            # alone.  Angular information is not lost: a path (l, l) -> 0 exists for
            # every l, so the previous layer's l=1 and l=2 features still feed in --
            # they are contracted to invariants here instead of being propagated and
            # then silently discarded (which would leave those weights gradient-less).
            last = layer == num_layers - 1
            irreps_out: Irreps = {0: hidden_multiplicity} if last else dict(irreps_hidden)
            self.convs.append(
                EquivariantConv(
                    irreps_in,
                    irreps_sh,
                    irreps_out,
                    num_radial=num_radial,
                    radial_hidden=radial_hidden,
                    avg_num_neighbors=avg_num_neighbors,
                )
            )
            self.gates.append(Gate(irreps_out))
            irreps_in = irreps_out

        # Readout sees only l=0, which is already invariant -- this is the single point
        # where equivariance is intentionally collapsed into invariance.
        self.readout = nn.Sequential(
            nn.Linear(hidden_multiplicity, hidden_multiplicity),
            nn.SiLU(),
            nn.Linear(hidden_multiplicity, 1),
        )

    def forward(
        self,
        species: Tensor,
        pos: Tensor,
        batch: Tensor | None = None,
        edge_index: Tensor | None = None,
    ) -> Tensor:
        """Predict one scalar per graph.

        ``species`` holds *indices* (0..num_species-1), not atomic numbers.
        """
        num_nodes = pos.shape[0]
        if batch is None:
            batch = pos.new_zeros(num_nodes, dtype=torch.long)
        if edge_index is None:
            edge_index = radius_graph(pos, self.cutoff, batch)

        src, dst = edge_index[0], edge_index[1]
        # Relative positions -- the only geometric quantity used, hence translation
        # invariance holds identically rather than approximately.
        edge_vec = pos[dst] - pos[src]
        edge_len = edge_vec.norm(dim=-1)

        sh: Features = {
            ell: spherical_harmonics_l(ell, edge_vec, normalization="component")
            for ell in self.sh_degrees
        }
        radial_features = self.radial_basis(edge_len)
        envelope = self.envelope(edge_len)

        x: Features = {0: self.embedding(species).unsqueeze(-1)}
        for conv, gate in zip(self.convs, self.gates, strict=True):
            x = gate(conv(x, sh, edge_index, radial_features, envelope, num_nodes))

        node_scalars = x[0].squeeze(-1)
        node_out = self.readout(node_scalars)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        return scatter_sum(node_out, batch, num_graphs).squeeze(-1)
