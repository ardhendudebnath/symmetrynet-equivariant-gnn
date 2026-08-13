r"""Phase 3: the full Tensor Field Network, built on ``e3nn``.

Same architecture as the hand-written :class:`~symmetrynet.scratch.layer.ScratchTFN`,
but using ``e3nn``'s optimised primitives so it can actually compete on QM9.  The
from-scratch version stays in the repository precisely because this one hides the
machinery: ``FullyConnectedTensorProduct`` is the same Clebsch-Gordan contraction
derived in :mod:`symmetrynet.scratch.clebsch_gordan`, just fused and compiled.

Layer structure (following Tensor Field Networks / NequIP):

.. code-block:: text

    x  --Linear-->  x'                        (mix channels within each degree)
    x'[src] (x) Y_l(r_hat)  weighted by MLP(|r|)   (create angular information)
       |
       +--> scatter-sum over neighbours / sqrt(avg_degree)
       |
       +  Linear(x)                           (self-interaction / skip)
       |
       --Gate-->  out                          (nonlinearity that preserves equivariance)

Parity bookkeeping
------------------
Irreps carry a parity label as well as a degree: spherical harmonics of degree
:math:`\ell` are ``0e, 1o, 2e, ...``, alternating because
:math:`Y_\ell(-\hat r) = (-1)^\ell Y_\ell(\hat r)`.  Matching the hidden irreps to that
pattern makes the model invariant under the *full* O(3) including inversion, which is
correct for a scalar quantity like the HOMO-LUMO gap.  ``e3nn`` enforces this
automatically: a path that would violate parity simply is not created.
"""

from __future__ import annotations

import math

import torch
from e3nn import o3
from e3nn.nn import BatchNorm, FullyConnectedNet, Gate
from torch import Tensor, nn

from ..nn.radial import BesselBasis, PolynomialCutoff
from ..utils.graph import radius_graph, scatter_sum

__all__ = ["TensorFieldNetwork", "TFNInteraction", "hidden_irreps"]


def hidden_irreps(multiplicity: int, l_max: int) -> o3.Irreps:
    """``mul x 0e + mul x 1o + mul x 2e + ...`` up to ``l_max``, with natural parity."""
    return o3.Irreps([(multiplicity, (ell, (-1) ** ell)) for ell in range(l_max + 1)])


def _build_gate(irreps_out: o3.Irreps) -> Gate:
    """Gated nonlinearity for ``irreps_out``.

    Scalars pass through SiLU.  Every non-scalar gets its own scalar gate, squashed by a
    sigmoid and multiplied in -- which is nonlinear yet commutes with :math:`D^\\ell`,
    the whole trick behind equivariant activations.
    """
    scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l == 0])
    gated = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l > 0])
    if len(gated) == 0:
        # l_max = 0 ablation: nothing to gate, so a plain activation suffices.
        return Gate(scalars, [torch.nn.functional.silu] * len(scalars), o3.Irreps(""), [], gated)
    gates = o3.Irreps([(mul, "0e") for mul, _ in gated])
    return Gate(
        scalars,
        [torch.nn.functional.silu] * len(scalars),
        gates,
        [torch.sigmoid] * len(gates),
        gated,
    )


def _build_uvu_tensor_product(
    irreps_in: o3.Irreps, irreps_sh: o3.Irreps, irreps_target: o3.Irreps
) -> tuple[o3.TensorProduct, o3.Irreps]:
    """Build a channel-wise (``uvu``) tensor product and its output irreps.

    Why not ``FullyConnectedTensorProduct``?  Because its weights are *per edge*, and a
    fully connected product learns one weight for every (input channel, output channel)
    pair on every path.  At ``multiplicity=64`` that is ~65k weights per edge; with the
    ~27k edges in a batch of 96 QM9 molecules the radial network would have to
    materialise a 7 GiB tensor for a single forward pass.  Measured, not guessed -- it
    is what an earlier version of this file did before it hit an OOM.

    ``uvu`` instead pairs input channel ``u`` with output channel ``u``, so each path
    carries ``mul`` weights rather than ``mul_in * mul_out``.  That is roughly a 100x
    reduction, and the cross-channel mixing that the fully connected version bought is
    recovered immediately afterwards by an ``o3.Linear`` -- which has *shared* weights
    and therefore costs nothing per edge.  This is the same decomposition NequIP uses.
    """
    target_irreps = {ir for _, ir in irreps_target}
    instructions: list[tuple[int, int, int, str, bool]] = []
    mid: list[tuple[int, o3.Irrep]] = []

    for i, (mul, ir_in) in enumerate(irreps_in):
        for j, (_, ir_sh) in enumerate(irreps_sh):
            for ir_out in ir_in * ir_sh:
                # Paths violating the selection rule or parity never appear here, so
                # the model cannot represent a parity-odd scalar even by accident.
                if ir_out in target_irreps:
                    instructions.append((i, j, len(mid), "uvu", True))
                    mid.append((mul, ir_out))

    if not instructions:
        raise ValueError(
            f"no equivariant path from {irreps_in} (x) {irreps_sh} into {irreps_target}"
        )

    irreps_mid = o3.Irreps(mid)
    tp = o3.TensorProduct(
        irreps_in,
        irreps_sh,
        irreps_mid,
        instructions,
        shared_weights=False,
        internal_weights=False,
    )
    return tp, irreps_mid


class TFNInteraction(nn.Module):
    """One equivariant convolution with a distance-conditioned tensor product."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_sh: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        num_radial: int,
        radial_hidden: int = 128,
        avg_num_neighbors: float = 12.0,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.gate = _build_gate(irreps_out)

        self.linear_in = o3.Linear(irreps_in, irreps_in)
        self.tp, irreps_mid = _build_uvu_tensor_product(irreps_in, irreps_sh, self.gate.irreps_in)
        # Restores the cross-channel mixing that `uvu` gives up, at shared-weight cost.
        self.linear_out = o3.Linear(irreps_mid, self.gate.irreps_in)

        # The radial network turns an invariant scalar (the distance) into the weights
        # of an equivariant operation.  This is what lets the interaction depend on
        # geometry without ever breaking the symmetry.
        self.radial = FullyConnectedNet(
            [num_radial, radial_hidden, radial_hidden, self.tp.weight_numel],
            torch.nn.functional.silu,
        )
        self.skip = o3.Linear(irreps_in, self.gate.irreps_in)
        self.irreps_out = self.gate.irreps_out

        # Equivariant normalisation of the residual stream.
        #
        # Dividing the neighbour sum by sqrt(avg_num_neighbors) assumes the incoming
        # messages are independent.  They are not -- messages arriving at one atom share
        # its central features and its local chemistry -- so the sum grows like N rather
        # than sqrt(N), leaving a systematic gain of order sqrt(N) ~ 4 per layer.
        # Measured across four layers, activations compounded 1.5 -> 4.7 -> 11.7 -> 47.9.
        #
        # e3nn's BatchNorm rescales each irrep by the *norm* of its components, which is
        # a rotation-invariant quantity, so equivariance is untouched (asserted in
        # tests/test_model_equivariance.py).
        self.norm = BatchNorm(self.irreps_out) if batch_norm else None

    def forward(
        self,
        x: Tensor,
        sh: Tensor,
        edge_index: Tensor,
        radial_features: Tensor,
        envelope: Tensor,
        num_nodes: int,
    ) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        weights = self.radial(radial_features) * envelope.unsqueeze(-1)
        messages = self.tp(self.linear_in(x)[src], sh, weights)
        aggregated = scatter_sum(messages, dst, num_nodes) / math.sqrt(self.avg_num_neighbors)
        out = self.gate(self.linear_out(aggregated) + self.skip(x))
        return self.norm(out) if self.norm is not None else out


class TensorFieldNetwork(nn.Module):
    """E(3)-equivariant message-passing network predicting one scalar per molecule."""

    def __init__(
        self,
        *,
        num_species: int = 5,
        multiplicity: int = 64,
        l_max: int = 2,
        num_layers: int = 4,
        cutoff: float = 5.0,
        num_radial: int = 8,
        radial_hidden: int = 128,
        avg_num_neighbors: float = 12.0,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.l_max = l_max
        self.irreps_sh = o3.Irreps.spherical_harmonics(l_max)

        self.embedding = nn.Embedding(num_species, multiplicity)
        self.radial_basis = BesselBasis(num_radial, cutoff)
        self.envelope = PolynomialCutoff(cutoff)

        irreps = o3.Irreps(f"{multiplicity}x0e")
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            # The readout reads scalars only, so the last layer produces scalars only.
            # Angular features still contribute: the (l, l) -> 0 paths contract them
            # into invariants here rather than propagating them to be discarded.
            last = layer == num_layers - 1
            irreps_out = (
                o3.Irreps(f"{multiplicity}x0e") if last else hidden_irreps(multiplicity, l_max)
            )
            block = TFNInteraction(
                irreps,
                self.irreps_sh,
                irreps_out,
                num_radial=num_radial,
                radial_hidden=radial_hidden,
                avg_num_neighbors=avg_num_neighbors,
                batch_norm=batch_norm,
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

        # 'component' normalisation keeps each harmonic's mean square at 1, so the
        # tensor product neither inflates nor shrinks activations layer over layer.
        sh = o3.spherical_harmonics(
            self.irreps_sh, edge_vec, normalize=True, normalization="component"
        )
        radial_features = self.radial_basis(edge_len)
        envelope = self.envelope(edge_len)

        x = self.embedding(species)
        for layer in self.layers:
            x = layer(x, sh, edge_index, radial_features, envelope, num_nodes)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        return scatter_sum(self.readout(x), batch, num_graphs).squeeze(-1)
