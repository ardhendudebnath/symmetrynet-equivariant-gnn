r"""Equivariant building blocks assembled from the hand-derived primitives.

Three operations suffice to build a Tensor Field Network layer, and each is equivariant
for a different reason:

``EquivariantLinear``
    Mixes *channels* within a fixed degree :math:`\ell`.  Equivariant because the weight
    matrix never touches the :math:`m` index, and :math:`D^{\ell}` acts only on :math:`m`.

``TensorProduct``
    Combines a degree-:math:`\ell_1` feature with a degree-:math:`\ell_2` feature into
    degree-:math:`\ell` outputs through the Clebsch-Gordan tensor.  Equivariant by the
    intertwining identity -- this is the only operation that can *create* new angular
    information.

``Gate``
    The nonlinearity.  Ordinary pointwise activations such as ReLU destroy equivariance
    for :math:`\ell > 0`, because :math:`\mathrm{ReLU}(D v) \ne D\,\mathrm{ReLU}(v)`.
    The fix is to only ever *scale* a higher-:math:`\ell` feature by an invariant
    (:math:`\ell = 0`) quantity, which commutes with :math:`D^{\ell}` trivially.

Feature convention
------------------
A node feature set is a ``dict`` mapping degree :math:`\ell` to a tensor of shape
``(num_nodes, multiplicity, 2*ell + 1)``.  Keeping the multiplicity and the :math:`m`
index as separate axes -- rather than flattening as ``e3nn`` does -- makes the
equivariance of every einsum below inspectable by eye.
"""

from __future__ import annotations

import math
from functools import cache

import torch
from torch import Tensor, nn

from .clebsch_gordan import clebsch_gordan, decomposition_degrees

__all__ = ["Irreps", "EquivariantLinear", "TensorProduct", "Gate"]


@cache
def _cg_for(l1: int, l2: int, l3: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Clebsch-Gordan tensor materialised directly in the requested precision.

    Deliberately *not* a registered buffer.  A buffer would be created once at some
    fixed dtype and then merely widened by ``module.to(torch.float64)`` -- silently
    keeping float32-truncated coefficients and capping equivariance error at ~1e-7.
    Building from the cached float64 source each time makes double-precision tests
    exact, and the cache keeps it free after the first call.
    """
    return clebsch_gordan(l1, l2, l3, dtype=dtype, device=device)


#: Degree -> multiplicity.  The lightweight stand-in for ``e3nn.o3.Irreps``.
Irreps = dict[int, int]

Features = dict[int, Tensor]


class EquivariantLinear(nn.Module):
    """Per-degree channel mixing: ``out[l][n, v, m] = sum_u W[l][v, u] * x[l][n, u, m]``.

    Note there is no bias for ``l > 0``: adding a constant to a vector feature would
    break equivariance, since a rotation acts on the feature but not on the constant.
    Scalars may carry a bias safely.
    """

    def __init__(self, irreps_in: Irreps, irreps_out: Irreps, *, bias: bool = True):
        super().__init__()
        self.irreps_in = dict(irreps_in)
        self.irreps_out = dict(irreps_out)
        self.weights = nn.ParameterDict()
        self.biases = nn.ParameterDict()
        for ell, mul_out in self.irreps_out.items():
            mul_in = self.irreps_in.get(ell)
            if mul_in is None:
                raise ValueError(f"output degree l={ell} has no matching input degree")
            # 1/sqrt(fan_in) keeps component variance ~1 through the layer.
            self.weights[str(ell)] = nn.Parameter(
                torch.randn(mul_out, mul_in) / math.sqrt(mul_in)
            )
            if bias and ell == 0:
                self.biases[str(ell)] = nn.Parameter(torch.zeros(mul_out, 1))

    def forward(self, x: Features) -> Features:
        out: Features = {}
        for ell, weight in self.weights.items():
            ell_i = int(ell)
            y = torch.einsum("vu,nui->nvi", weight, x[ell_i])
            if ell in self.biases:
                y = y + self.biases[ell]
            out[ell_i] = y
        return out


class TensorProduct(nn.Module):
    r"""Clebsch-Gordan tensor product with externally supplied (per-edge) weights.

    Computes, for every allowed path :math:`(\ell_1, \ell_2) \to \ell`,

    .. math::
        \mathrm{out}^{\ell}_{v,k} \;=\;
        \sum_{u} w^{(\ell_1 \ell_2 \ell)}_{v u}
        \sum_{i j} C^{(\ell_1 \ell_2 \ell)}_{ijk}\, x^{\ell_1}_{u,i}\, y^{\ell_2}_{j},

    and sums the contributions to each output degree.

    The weights are *not* parameters of this module.  In a Tensor Field Network they are
    produced by a small MLP applied to the interatomic distance -- that is what makes the
    interaction depend on how far apart two atoms are while remaining exactly equivariant,
    since a distance is itself invariant.

    The ``x`` argument carries a multiplicity axis; ``y`` (the spherical harmonics of the
    edge direction) does not, since there is only one geometry per edge.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_sh: Irreps,
        irreps_out: Irreps,
        *,
        normalize_paths: bool = True,
    ):
        super().__init__()
        self.irreps_in = dict(irreps_in)
        self.irreps_sh = dict(irreps_sh)
        self.irreps_out = dict(irreps_out)
        self.normalize_paths = normalize_paths

        # Enumerate the selection-rule-allowed paths once, at construction time.
        paths: list[tuple[int, int, int]] = []
        for l_in in sorted(self.irreps_in):
            for l_sh in sorted(self.irreps_sh):
                for l_out in decomposition_degrees(l_in, l_sh):
                    if l_out in self.irreps_out:
                        paths.append((l_in, l_sh, l_out))
        if not paths:
            raise ValueError(
                "no CG path connects the requested irreps; check the selection rule "
                f"|l1-l2| <= l <= l1+l2 for in={sorted(self.irreps_in)}, "
                f"sh={sorted(self.irreps_sh)}, out={sorted(self.irreps_out)}"
            )
        self.paths = paths

        # Per-output-degree path count, used to keep the summed variance ~1.
        self._paths_per_out: dict[int, int] = {}
        for _, _, l_out in paths:
            self._paths_per_out[l_out] = self._paths_per_out.get(l_out, 0) + 1

    @property
    def weight_numel(self) -> int:
        """Total number of per-edge weights the radial network must produce."""
        return sum(
            self.irreps_out[l_out] * self.irreps_in[l_in] for l_in, _, l_out in self.paths
        )

    def path_weight_shapes(self) -> list[tuple[int, int]]:
        return [(self.irreps_out[l_out], self.irreps_in[l_in]) for l_in, _, l_out in self.paths]

    def forward(self, x: Features, sh: Features, weights: Tensor) -> Features:
        """
        Parameters
        ----------
        x:
            ``{l: (E, mul_in, 2l+1)}`` -- input features, already gathered per edge.
        sh:
            ``{l: (E, 2l+1)}`` -- spherical harmonics of each edge direction.
        weights:
            ``(E, weight_numel)`` -- flat per-edge weights from the radial network.

        Returns
        -------
        ``{l: (E, mul_out, 2l+1)}``
        """
        if weights.shape[-1] != self.weight_numel:
            raise ValueError(
                f"expected {self.weight_numel} weights per edge, got {weights.shape[-1]}"
            )

        num_edges = weights.shape[0]
        out: Features = {
            l_out: weights.new_zeros(num_edges, mul, 2 * l_out + 1)
            for l_out, mul in self.irreps_out.items()
        }

        offset = 0
        for (l_in, l_sh, l_out), (mul_out, mul_in) in zip(
            self.paths, self.path_weight_shapes(), strict=True
        ):
            size = mul_out * mul_in
            w = weights[:, offset : offset + size].reshape(num_edges, mul_out, mul_in)
            offset += size

            cg = _cg_for(l_in, l_sh, l_out, weights.dtype, weights.device)
            # e: edge, v: out channel, u: in channel, i/j/k: the m indices
            contrib = torch.einsum("evu,eui,ej,ijk->evk", w, x[l_in], sh[l_sh], cg)
            if self.normalize_paths:
                # Summing mul_in channels and several paths each inflate the variance.
                contrib = contrib / math.sqrt(mul_in * self._paths_per_out[l_out])
            out[l_out] = out[l_out] + contrib

        return out


class Gate(nn.Module):
    r"""Gated nonlinearity: activate scalars, and *scale* higher degrees by invariants.

    For :math:`\ell = 0` we apply ``act`` directly -- legitimate because
    :math:`D^{0} = 1`, so any function of a scalar stays invariant.

    For :math:`\ell > 0` we compute a gate :math:`g` as a learned function of the node's
    scalar channels and return :math:`g \odot x^{\ell}`, broadcasting over :math:`m`.
    Since :math:`g` is invariant, :math:`g \odot (D^{\ell} x) = D^{\ell} (g \odot x)`, so
    equivariance survives while the layer still gets a genuine nonlinearity.

    (``e3nn``'s ``Gate`` instead routes extra scalar channels out of the tensor product
    itself.  Both are equivariant; deriving the gates from the existing scalars keeps
    this from-scratch version readable.)
    """

    def __init__(
        self,
        irreps: Irreps,
        *,
        act: nn.Module | None = None,
        gate_act: nn.Module | None = None,
    ):
        super().__init__()
        self.irreps = dict(irreps)
        if 0 not in self.irreps:
            raise ValueError("Gate needs l=0 channels to build gates from")
        self.act = act or nn.SiLU()
        self.gate_act = gate_act or nn.Sigmoid()

        higher = {ell: mul for ell, mul in self.irreps.items() if ell > 0}
        self.higher_degrees = sorted(higher)
        num_gates = sum(higher.values())
        self.gate_proj = (
            nn.Linear(self.irreps[0], num_gates) if num_gates > 0 else None
        )

    def forward(self, x: Features) -> Features:
        out: Features = {0: self.act(x[0])}
        if self.gate_proj is None:
            return out

        scalars = x[0].squeeze(-1)  # (N, mul_0)
        gates = self.gate_act(self.gate_proj(scalars))  # (N, num_gates)

        offset = 0
        for ell in self.higher_degrees:
            mul = self.irreps[ell]
            g = gates[:, offset : offset + mul].unsqueeze(-1)  # (N, mul, 1)
            offset += mul
            out[ell] = x[ell] * g
        return out
