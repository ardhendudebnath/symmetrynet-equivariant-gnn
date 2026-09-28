r"""NequIP: the strongest available challenge to this project's own conclusion.

Batzner et al., *E(3)-equivariant graph neural networks for data-efficient and accurate
interatomic potentials* (2022).

Why this model is here
----------------------
The results so far say that Clebsch-Gordan tensor products cost roughly 5x the compute of
PaiNN's vector algebra and do not repay it: the TFN loses to a distance-only baseline, and
per-component gradient sensitivity to :math:`\ell = 2` features is ~3.7x lower than to
scalars. The obvious objection is that the TFN is a 2018 design, and that a *well-built*
CG-based model would tell a different story. NequIP is that model, and it is the honest
way to try to break the conclusion rather than defend it.

What NequIP changes, relative to :mod:`symmetrynet.models.tfn`
---------------------------------------------------------------
Three differences, and none of them is simply "bigger":

**Species enter at every layer, through a tensor product.** The TFN embeds atomic species
once at the input and then never refers to them again. NequIP's self-connection is a
``FullyConnectedTensorProduct`` between the node features and the species one-hot, applied
in *every* interaction block. Chemistry is therefore re-injected at each depth rather than
having to survive the whole network inside the features. The self-connection carries shared
weights and is evaluated per node, so unlike the per-edge fully connected product that
needed 7.1 GiB in an earlier version of the TFN, this one is cheap.

**Both parities are carried.** The TFN uses natural parity only -- ``0e, 1o, 2e`` -- so its
features are exactly true tensors. NequIP also carries ``0o, 1e, 2o``, which opens tensor
product paths that natural parity forbids. Odd scalars need an odd activation to stay
equivariant, which is why the gate below dispatches ``tanh`` for ``0o`` and ``SiLU`` for
``0e``; using SiLU on an odd scalar would silently break parity, and ``e3nn`` rejects it.

**Ordering around the convolution.** Linear, then convolution, then linear, then add the
self-connection, then gate. The TFN applies its skip before the final linear. Minor on
paper, but it changes what the gate sees.

Everything else is deliberately shared with the rest of the project -- same radial basis,
cutoff, readout, aggregation normalisation and training loop -- so that a difference in
accuracy is attributable to the architecture rather than to tuning.
"""

from __future__ import annotations

import math

import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from torch import Tensor, nn

from ..nn.radial import BesselBasis, PolynomialCutoff
from ..utils.graph import radius_graph, scatter_sum
from .tfn import _build_uvu_tensor_product

__all__ = ["NequIP", "NequIPInteraction", "nequip_irreps"]


def nequip_irreps(multiplicity: int, l_max: int) -> o3.Irreps:
    r"""``mul x {0e,0o,1e,1o,...}`` up to ``l_max`` -- both parities at every degree.

    This is the representational difference from the TFN, which carries only the natural
    parity :math:`(-1)^{\ell}`. Including the opposite parity roughly doubles the width at
    each degree and, more importantly, opens paths that natural parity forbids.
    """
    entries = []
    for ell in range(l_max + 1):
        for parity in (1, -1):
            entries.append((multiplicity, (ell, parity)))
    return o3.Irreps(entries)


def _build_gate(irreps_out: o3.Irreps) -> Gate:
    """Gated nonlinearity that respects parity.

    An even scalar may use any activation. An **odd** scalar may not: applying SiLU to a
    quantity that must flip sign under inversion destroys the parity it is supposed to
    carry. ``tanh`` is odd, so it is safe. ``e3nn`` validates this and raises rather than
    letting it pass silently, which is the behaviour you want.
    """
    scalars, acts = [], []
    for mul, ir in irreps_out:
        if ir.l != 0:
            continue
        scalars.append((mul, ir))
        acts.append(torch.tanh if ir.p == -1 else torch.nn.functional.silu)
    irreps_scalars = o3.Irreps(scalars)

    gated = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l > 0])
    if len(gated) == 0:
        return Gate(irreps_scalars, acts, o3.Irreps(""), [], gated)

    # Gates are even scalars: multiplying by an even scalar leaves parity untouched.
    irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in gated])
    return Gate(
        irreps_scalars, acts, irreps_gates, [torch.sigmoid] * len(irreps_gates), gated
    )


class NequIPInteraction(nn.Module):
    """One NequIP interaction block."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_sh: o3.Irreps,
        irreps_out: o3.Irreps,
        irreps_species: o3.Irreps,
        *,
        num_radial: int,
        radial_hidden: int = 64,
        avg_num_neighbors: float = 12.0,
    ):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.gate = _build_gate(irreps_out)

        self.linear_in = o3.Linear(irreps_in, irreps_in)
        self.tp, irreps_mid = _build_uvu_tensor_product(
            irreps_in, irreps_sh, self.gate.irreps_in
        )
        self.linear_out = o3.Linear(irreps_mid, self.gate.irreps_in)

        # Distance is invariant, so it may parameterise an equivariant operation freely.
        self.radial = FullyConnectedNet(
            [num_radial, radial_hidden, radial_hidden, self.tp.weight_numel],
            torch.nn.functional.silu,
        )

        # The self-connection: node features combined with the species one-hot. Shared
        # weights, evaluated per node, so the fully connected product is affordable here
        # even though it was not for the per-edge convolution.
        self.self_connection = o3.FullyConnectedTensorProduct(
            irreps_in, irreps_species, self.gate.irreps_in
        )
        self.irreps_out = self.gate.irreps_out

    def forward(
        self,
        x: Tensor,
        species_onehot: Tensor,
        sh: Tensor,
        edge_index: Tensor,
        radial: Tensor,
        envelope: Tensor,
        num_nodes: int,
    ) -> Tensor:
        src, dst = edge_index[0], edge_index[1]

        # Computed before the convolution overwrites x, and re-injects chemistry at depth.
        skip = self.self_connection(x, species_onehot)

        weights = self.radial(radial) * envelope.unsqueeze(-1)
        messages = self.tp(self.linear_in(x)[src], sh, weights)
        aggregated = scatter_sum(messages, dst, num_nodes) / math.sqrt(self.avg_num_neighbors)

        return self.gate(self.linear_out(aggregated) + skip)


class NequIP(nn.Module):
    """E(3)-equivariant interatomic-potential network, predicting one scalar per molecule."""

    def __init__(
        self,
        *,
        num_species: int = 5,
        multiplicity: int = 32,
        l_max: int = 2,
        num_layers: int = 4,
        cutoff: float = 5.0,
        num_radial: int = 8,
        radial_hidden: int = 64,
        avg_num_neighbors: float = 12.0,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.num_species = num_species
        self.irreps_sh = o3.Irreps.spherical_harmonics(l_max)
        self.irreps_species = o3.Irreps(f"{num_species}x0e")

        # The first features are the species one-hot itself, which is what the initial
        # self-connection then mixes with.
        self.embedding = o3.Linear(self.irreps_species, o3.Irreps(f"{multiplicity}x0e"))

        self.radial_basis = BesselBasis(num_radial, cutoff)
        self.envelope = PolynomialCutoff(cutoff)

        irreps = o3.Irreps(f"{multiplicity}x0e")
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            # The readout reads scalars only, so the last block emits scalars only; the
            # (l, l) -> 0 paths still contract the angular features into invariants.
            last = layer == num_layers - 1
            irreps_out = (
                o3.Irreps(f"{multiplicity}x0e")
                if last
                else nequip_irreps(multiplicity, l_max)
            )
            block = NequIPInteraction(
                irreps,
                self.irreps_sh,
                irreps_out,
                self.irreps_species,
                num_radial=num_radial,
                radial_hidden=radial_hidden,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.layers.append(block)
            irreps = block.irreps_out

        self.readout = nn.Sequential(
            nn.Linear(multiplicity, multiplicity // 2),
            nn.SiLU(),
            nn.Linear(multiplicity // 2, 1),
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

        sh = o3.spherical_harmonics(
            self.irreps_sh, edge_vec, normalize=True, normalization="component"
        )
        radial = self.radial_basis(edge_len)
        envelope = self.envelope(edge_len)

        species_onehot = torch.nn.functional.one_hot(species, self.num_species).to(pos.dtype)
        x = self.embedding(species_onehot)

        for layer in self.layers:
            x = layer(x, species_onehot, sh, edge_index, radial, envelope, num_nodes)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        return scatter_sum(self.readout(x), batch, num_graphs).squeeze(-1)
