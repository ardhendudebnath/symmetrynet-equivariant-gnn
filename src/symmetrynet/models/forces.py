r"""Force prediction as the negative gradient of a predicted energy.

Why not simply add three output channels?
-----------------------------------------
A network could emit a 3-vector per atom directly, but predicting forces as
:math:`F_i = -\partial E / \partial x_i` buys three properties that a direct head does not:

**Equivariance for free.** :math:`E` is invariant, so its gradient with respect to
positions is automatically an :math:`\ell = 1` object transforming as
:math:`F(Rx) = R\,F(x)`. Any invariant energy model becomes a correctly equivariant force
model with no new machinery -- which is worth stating plainly, because it means the
*baseline* also produces equivariant forces. The force experiment is therefore not
"equivariant vs not"; both are equivariant. It compares how well each represents a
tensorial target.

**Energy conservation.** A conservative force field is by definition the gradient of a
potential. A direct vector head can predict a field with non-zero curl, which is
unphysical and shows up as drift in molecular dynamics.

**Newton's third law.** Since :math:`E` depends only on relative positions, the forces sum
to zero automatically. :func:`ForceModel.check_force_conservation` asserts this rather
than assuming it.

The cost is a double backward: obtaining :math:`\partial E/\partial x` inside the training
step means the graph must be retained so the loss can then be differentiated with respect
to the weights. That roughly doubles step time and is why ``create_graph`` is set from the
model's training mode rather than hard-coded.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["ForceModel"]


class ForceModel(nn.Module):
    """Wraps any energy model so it also returns forces via autograd.

    The wrapped model must map ``(species, pos, batch)`` to one scalar per graph, which is
    the interface every model in :mod:`symmetrynet.models` already provides.
    """

    def __init__(self, energy_model: nn.Module):
        super().__init__()
        self.energy_model = energy_model

    def forward(
        self,
        species: Tensor,
        pos: Tensor,
        batch: Tensor | None = None,
        *,
        create_graph: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(energy_per_graph, forces_per_atom)``.

        ``create_graph`` defaults to ``self.training``: it must be true while training so
        the force loss is itself differentiable, and false at evaluation to avoid building
        a graph that is never used.
        """
        if create_graph is None:
            create_graph = self.training

        # grad_mode must be forced on even under torch.no_grad(), or evaluation would
        # silently return zero forces -- a failure that looks like a badly trained model
        # rather than a bug.
        with torch.enable_grad():
            pos = pos.requires_grad_(True)
            energy = self.energy_model(species, pos, batch)
            (grad,) = torch.autograd.grad(
                energy.sum(),
                pos,
                create_graph=create_graph,
                retain_graph=create_graph,
            )
        forces = -grad
        return energy, forces if create_graph else forces.detach()

    @torch.no_grad()
    def check_force_conservation(
        self,
        species: Tensor,
        pos: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        r"""Net force per molecule, which must be zero.

        Because the energy depends only on relative positions, translating a molecule
        cannot change it, so :math:`\sum_i F_i = 0` identically. A non-zero value means the
        model is reading absolute coordinates somewhere.
        """
        from ..utils.graph import scatter_sum

        if batch is None:
            batch = pos.new_zeros(pos.shape[0], dtype=torch.long)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        _, forces = self(species, pos, batch, create_graph=False)
        return scatter_sum(forces, batch, num_graphs)
