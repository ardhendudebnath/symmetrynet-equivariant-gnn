"""Phase 4: the equivariance verification suite.

The project's correctness metric is ``||f(Rx) - f(x)||`` over many random rotations.  For
the equivariant models this must sit at floating-point noise; for the raw-coordinate
control it must visibly fail.  Both directions are asserted, because a model that
returns a constant would pass the first test trivially.

Two methodological points these tests encode:

1. **Everything runs in float64, and e3nn models are *constructed* in float64.**
   See :mod:`symmetrynet.utils.precision` -- casting after construction leaves
   float32 Wigner-3j constants baked in and inflates the residual by ~1e6 for
   ``l_max >= 2``.
2. **The residual is compared against the model's own reproducibility floor.**
   An "equivariance error" below the level at which the model reproduces itself is not
   measuring symmetry at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from symmetrynet.models import (
    InvariantGNN,
    NaiveCoordinateGNN,
    NaiveCoordinateMLP,
    PaiNN,
    ScratchTFN,
    TensorFieldNetwork,
)
from symmetrynet.scratch.wigner import random_rotation
from symmetrynet.utils.precision import default_dtype

DT = torch.float64
NUM_ROTATIONS = 25

#: Residual we require of a genuinely equivariant model in float64.
EXACT_TOL = 1e-11
#: Residual below which the naive control would *not* count as broken.
BROKEN_FLOOR = 1e-4


def build(name: str, **kwargs):
    """Construct a model in float64 so e3nn bakes exact constants."""
    with default_dtype(DT):
        builders = {
            "baseline": lambda: InvariantGNN(hidden=32, num_layers=3, **kwargs),
            "tfn_l0": lambda: TensorFieldNetwork(multiplicity=16, l_max=0, num_layers=3, **kwargs),
            "tfn_l1": lambda: TensorFieldNetwork(multiplicity=16, l_max=1, num_layers=3, **kwargs),
            "tfn_l2": lambda: TensorFieldNetwork(multiplicity=16, l_max=2, num_layers=3, **kwargs),
            "painn": lambda: PaiNN(hidden=32, num_layers=3, **kwargs),
            "scratch": lambda: ScratchTFN(hidden_multiplicity=8, l_max=2, num_layers=2, **kwargs),
            "naive": lambda: NaiveCoordinateGNN(hidden=32, num_layers=3, **kwargs),
            "naive_mlp": lambda: NaiveCoordinateMLP(hidden=64, **kwargs),
        }
        return builders[name]().eval()


EQUIVARIANT = ["baseline", "tfn_l0", "tfn_l1", "tfn_l2", "painn", "scratch"]
NOT_EQUIVARIANT = ["naive", "naive_mlp"]


def rotation_residual(model, species, pos, batch, trials=NUM_ROTATIONS) -> float:
    with torch.no_grad():
        base = model(species, pos, batch)
        worst = 0.0
        for _ in range(trials):
            rot = random_rotation(1, dtype=DT)
            worst = max(worst, (model(species, pos @ rot.T, batch) - base).abs().max().item())
    return worst


# --------------------------------------------------------------------------- exact
@pytest.mark.parametrize("name", EQUIVARIANT)
def test_rotation_invariance_is_exact(name, molecules):
    species, pos, batch, _ = molecules
    assert rotation_residual(build(name), species, pos, batch) < EXACT_TOL


@pytest.mark.parametrize("name", EQUIVARIANT)
def test_translation_invariance_is_exact(name, molecules):
    species, pos, batch, _ = molecules
    model = build(name)
    with torch.no_grad():
        base = model(species, pos, batch)
        for _ in range(10):
            shift = torch.randn(3, dtype=DT) * 30.0
            assert (model(species, pos + shift, batch) - base).abs().max() < 1e-9


@pytest.mark.parametrize("name", EQUIVARIANT)
def test_o3_invariance_including_inversion(name, molecules):
    """Parity too: for scalar targets the model should be invariant under all of O(3)."""
    species, pos, batch, _ = molecules
    model = build(name)
    with torch.no_grad():
        base = model(species, pos, batch)
        for _ in range(10):
            improper = -random_rotation(1, dtype=DT)  # det = -1
            assert (model(species, pos @ improper.T, batch) - base).abs().max() < EXACT_TOL


@pytest.mark.parametrize("name", EQUIVARIANT + NOT_EQUIVARIANT)
def test_permutation_invariance(name, molecules):
    """True of every model here -- it comes from sum pooling, not from equivariance."""
    species, pos, batch, sizes = molecules
    model = build(name)
    offsets = torch.tensor(sizes).cumsum(0) - torch.tensor(sizes)
    perm = torch.cat(
        [torch.randperm(n) + off for n, off in zip(sizes, offsets.tolist(), strict=True)]
    )
    with torch.no_grad():
        got = model(species[perm], pos[perm], batch)
        assert torch.allclose(got, model(species, pos, batch), atol=1e-10)


# ------------------------------------------------------------------------ controls
@pytest.mark.parametrize("name", NOT_EQUIVARIANT)
def test_naive_models_visibly_break_under_rotation(name, molecules):
    """The negative control must actually fail, or the positive result means nothing."""
    species, pos, batch, _ = molecules
    residual = rotation_residual(build(name), species, pos, batch)
    assert residual > BROKEN_FLOOR, (
        f"{name} was expected to violate rotation invariance but only moved by {residual:.2e}"
    )


@pytest.mark.parametrize("name", EQUIVARIANT)
def test_model_is_not_trivially_constant(name, molecules):
    """Guards against passing the invariance tests by ignoring the input."""
    species, pos, batch, _ = molecules
    model = build(name)
    with torch.no_grad():
        base = model(species, pos, batch)
        perturbed = model(species, pos + torch.randn_like(pos) * 0.5, batch)
    assert (perturbed - base).abs().max() > 1e-5


@pytest.mark.parametrize("name", EQUIVARIANT)
def test_residual_is_far_below_the_naive_control(name, molecules):
    """The headline claim, asserted directly: many orders of magnitude of separation."""
    species, pos, batch, _ = molecules
    equivariant = rotation_residual(build(name), species, pos, batch)
    naive = rotation_residual(build("naive"), species, pos, batch)
    assert naive / max(equivariant, 1e-16) > 1e6


# ----------------------------------------------------------- construction precision
def test_float64_construction_matters_for_high_l():
    """Documents the e3nn gotcha as an executable fact rather than a comment.

    Casting a float32-constructed model to float64 leaves truncated Wigner-3j constants
    baked into the tensor product, inflating the residual by orders of magnitude for
    ``l_max >= 2``.  If a future e3nn release fixes this, the test fails loudly and the
    note in ``symmetrynet.utils.precision`` can be retired.
    """
    torch.manual_seed(0)
    sizes = [10, 8]
    species = torch.cat([torch.randint(0, 5, (n,)) for n in sizes])
    pos = torch.cat([torch.randn(n, 3, dtype=DT) * 1.6 for n in sizes])
    batch = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(sizes)])

    torch.manual_seed(0)
    cast_after = TensorFieldNetwork(multiplicity=16, l_max=2, num_layers=3).double().eval()
    torch.manual_seed(0)
    with default_dtype(DT):
        built_in = TensorFieldNetwork(multiplicity=16, l_max=2, num_layers=3).eval()

    residual_cast = rotation_residual(cast_after, species, pos, batch, trials=10)
    residual_built = rotation_residual(built_in, species, pos, batch, trials=10)

    assert residual_built < EXACT_TOL
    assert residual_cast > residual_built * 100


def test_painn_vector_features_are_genuinely_equivariant(molecules):
    """PaiNN's internal vectors must *rotate*, not merely leave the output invariant.

    An invariant scalar output is a weak test on its own: a model whose vector channels
    silently collapsed to zero, or which never fed them into the readout, would still
    pass it while being a distance-only network wearing a costume. Here we rotate the
    input and require the hidden vector features themselves to come back rotated by the
    same matrix -- and separately require them to be non-trivial.
    """
    species, pos, batch, _ = molecules
    model = build("painn")

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["v"] = output[1].detach().clone()

    handle = model.updates[-1].register_forward_hook(hook)
    with torch.no_grad():
        model(species, pos, batch)
        v_base = captured["v"]

        rot = random_rotation(1, dtype=DT)
        model(species, pos @ rot.T, batch)
        v_rot = captured["v"]
    handle.remove()

    # v has shape (N, 3, F); the rotation acts on the spatial axis only.
    expected = torch.einsum("ij,njf->nif", rot, v_base)
    assert torch.allclose(v_rot, expected, atol=1e-11)

    # And the vectors must actually carry information.
    assert v_base.abs().max() > 1e-6, "vector channels collapsed to zero"


def test_painn_uses_no_tensor_products():
    """The point of PaiNN is that vector algebra suffices -- assert it stays that way.

    Checks the module's *imports* via the AST rather than grepping the text, because the
    docstring quite reasonably discusses e3nn and Clebsch-Gordan products while the code
    must not depend on them.
    """
    source = Path(__file__).resolve().parents[1] / "src/symmetrynet/models/painn.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)

    banned = ("e3nn", "wigner", "clebsch", "spherical_harmonics", "tensor_product")
    offenders = [
        name for name in imported if any(token in name.lower() for token in banned)
    ]
    assert not offenders, f"painn.py should not import {offenders}"


def test_gpu_reproducibility_floor_is_below_tolerance(molecules):
    """The equivariance tolerance must be meaningful, not swamped by scatter-add noise."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    species, pos, batch, _ = molecules
    model = build("tfn_l2").cuda()
    species, pos, batch = species.cuda(), pos.cuda(), batch.cuda()
    with torch.no_grad():
        a = model(species, pos, batch)
        floor = max((model(species, pos, batch) - a).abs().max().item() for _ in range(5))
    assert floor < EXACT_TOL
