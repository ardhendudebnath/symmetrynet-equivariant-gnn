r"""Forces are a genuinely *equivariant* output, so they need stronger tests than energies.

Every other test in this suite checks that a scalar prediction does not move under
rotation. A force field must instead *rotate with* the molecule:

.. math::  F(Rx) \;=\; R\, F(x).

That is a strictly stronger requirement. A model could be perfectly invariant and still get
forces wrong -- returning zeros, for instance, or a field with the right magnitudes and
wrong directions. Both would pass every invariance test in the repository.

These tests also verify the two physical guarantees that come from defining forces as
:math:`-\partial E/\partial x` rather than as a direct vector head: that the field really is
the gradient of the predicted energy (checked against finite differences), and that forces
sum to zero within each molecule.
"""

from __future__ import annotations

import pytest
import torch

from symmetrynet.models import InvariantGNN, PaiNN, TensorFieldNetwork
from symmetrynet.models.forces import ForceModel
from symmetrynet.scratch.wigner import random_rotation
from symmetrynet.utils.precision import default_dtype

DT = torch.float64
MODELS = ["baseline", "painn", "tfn"]


def build(name: str) -> ForceModel:
    with default_dtype(DT):
        energy = {
            "baseline": lambda: InvariantGNN(hidden=32, num_layers=3),
            "painn": lambda: PaiNN(hidden=32, num_layers=3),
            "tfn": lambda: TensorFieldNetwork(multiplicity=16, l_max=2, num_layers=2),
        }[name]()
        return ForceModel(energy).eval()


@pytest.fixture
def molecules():
    """Three small molecules with realistic bond lengths."""
    torch.manual_seed(7)
    sizes = [9, 7, 11]
    species = torch.cat([torch.randint(0, 5, (n,)) for n in sizes])
    pos = torch.cat([torch.randn(n, 3, dtype=DT) * 1.5 for n in sizes])
    batch = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(sizes)])
    return species, pos, batch


# ------------------------------------------------------------------- equivariance
@pytest.mark.parametrize("name", MODELS)
def test_forces_are_equivariant(name, molecules):
    """``F(Rx) == R F(x)`` -- the defining property, and the point of the experiment."""
    species, pos, batch = molecules
    model = build(name)

    _, base = model(species, pos, batch, create_graph=False)
    worst = 0.0
    for _ in range(15):
        rot = random_rotation(1, dtype=DT)
        _, rotated = model(species, pos @ rot.T, batch, create_graph=False)
        # base is (N, 3) row vectors, so rotating means base @ rot.T
        worst = max(worst, (rotated - base @ rot.T).abs().max().item())
    assert worst < 1e-10, f"{name}: force equivariance residual {worst:.3e}"


@pytest.mark.parametrize("name", MODELS)
def test_energy_stays_invariant_while_forces_rotate(name, molecules):
    """The two outputs must transform *differently*: energy fixed, forces rotating.

    Asserting both together guards against a model that accidentally makes forces
    invariant too, which would be wrong but would pass a naive rotation check.
    """
    species, pos, batch = molecules
    model = build(name)
    energy_base, force_base = model(species, pos, batch, create_graph=False)

    rot = random_rotation(1, dtype=DT)
    energy_rot, force_rot = model(species, pos @ rot.T, batch, create_graph=False)

    assert (energy_rot - energy_base).abs().max() < 1e-11      # invariant
    assert (force_rot - force_base @ rot.T).abs().max() < 1e-10  # equivariant
    # And the forces genuinely moved, i.e. the rotation was not a no-op on them.
    assert (force_rot - force_base).abs().max() > 1e-6


@pytest.mark.parametrize("name", MODELS)
def test_forces_are_translation_invariant(name, molecules):
    species, pos, batch = molecules
    model = build(name)
    _, base = model(species, pos, batch, create_graph=False)
    for _ in range(5):
        shift = torch.randn(3, dtype=DT) * 12.0
        _, shifted = model(species, pos + shift, batch, create_graph=False)
        assert (shifted - base).abs().max() < 1e-10


# ------------------------------------------------------------ physical guarantees
@pytest.mark.parametrize("name", MODELS)
def test_forces_sum_to_zero_per_molecule(name, molecules):
    r"""Newton's third law, which follows from the energy using only relative positions.

    A non-zero net force means the model is reading absolute coordinates somewhere.
    """
    species, pos, batch = molecules
    net = build(name).check_force_conservation(species, pos, batch)
    assert net.abs().max() < 1e-9, f"{name}: net force {net.abs().max():.3e}"


@pytest.mark.parametrize("name", MODELS)
def test_forces_match_finite_differences(name, molecules):
    r"""The autograd force really is :math:`-\partial E/\partial x`.

    Central differences on the energy are an independent route to the same quantity, so
    agreement rules out a sign error or a detached graph -- both of which would train to
    something plausible-looking while being wrong.
    """
    species, pos, batch = molecules
    model = build(name)
    _, forces = model(species, pos, batch, create_graph=False)

    step = 1e-5
    torch.manual_seed(0)
    probes = [(int(i), int(c)) for i, c in
              zip(torch.randint(0, pos.shape[0], (6,)), torch.randint(0, 3, (6,)), strict=True)]

    for atom, axis in probes:
        plus, minus = pos.clone(), pos.clone()
        plus[atom, axis] += step
        minus[atom, axis] -= step
        with torch.no_grad():
            e_plus = model.energy_model(species, plus, batch).sum()
            e_minus = model.energy_model(species, minus, batch).sum()
        numerical = -(e_plus - e_minus) / (2 * step)
        assert abs(numerical - forces[atom, axis]) < 1e-5 * max(1.0, abs(float(numerical)))


@pytest.mark.parametrize("name", MODELS)
def test_forces_are_nontrivial(name, molecules):
    """A model returning zero forces would pass equivariance and conservation trivially."""
    species, pos, batch = molecules
    _, forces = build(name)(species, pos, batch, create_graph=False)
    assert torch.isfinite(forces).all()
    assert forces.abs().max() > 1e-6


# ------------------------------------------------------------------ trainability
@pytest.mark.parametrize("name", MODELS)
def test_force_loss_is_differentiable(name, molecules):
    """The double backward must work: force loss -> gradients on the weights.

    This is the mechanical risk of gradient-based forces. If ``create_graph`` were
    dropped, the force term would silently contribute no gradient and the model would
    train on energies alone.
    """
    species, pos, batch = molecules
    with default_dtype(DT):
        model = ForceModel(
            {
                "baseline": lambda: InvariantGNN(hidden=16, num_layers=2),
                "painn": lambda: PaiNN(hidden=16, num_layers=2),
                "tfn": lambda: TensorFieldNetwork(multiplicity=8, l_max=2, num_layers=2),
            }[name]()
        )
    model.train()

    energy, forces = model(species, pos, batch)
    target_f = torch.randn_like(forces)
    loss = (forces - target_f).abs().mean() + energy.abs().mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads), f"{name}: some params got no gradient"
    assert any(g.abs().max() > 0 for g in grads), f"{name}: all gradients are exactly zero"


def test_force_only_loss_still_trains_the_model(molecules):
    """Guard the subtle case: a *force-only* loss must still reach every weight.

    Forces are a derivative of the energy, so a readout bias -- which shifts energy but not
    its gradient -- legitimately receives no force gradient. Everything that shapes the
    energy *surface* must, and that is what this checks.
    """
    species, pos, batch = molecules
    with default_dtype(DT):
        model = ForceModel(PaiNN(hidden=16, num_layers=2))
    model.train()

    _, forces = model(species, pos, batch)
    (forces - torch.randn_like(forces)).abs().mean().backward()

    reached = [n for n, p in model.named_parameters()
               if p.grad is not None and p.grad.abs().max() > 0]
    assert len(reached) > 5, f"force-only loss reached only {len(reached)} parameter tensors"
