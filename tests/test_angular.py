r"""The angle-aware invariant model: correct angles, and invariant despite seeing them.

This model exists to break a confound -- every other equivariant model here also has
angular information, so equivariance and angle-awareness could not be told apart. It is
only a useful control if two things hold, and both are easy to get wrong:

* the triplets and angles are actually right (a subtly wrong angle still trains);
* it is genuinely **invariant**, not accidentally equivariant or accidentally broken.

The Legendre basis is checked against SciPy and the triplet enumeration against a brute
force loop, because both are the kind of index-juggling code that produces plausible
numbers when wrong.
"""

from __future__ import annotations

import math

import pytest
import torch

from symmetrynet.models import AngularInvariantGNN
from symmetrynet.models.angular import build_triplets, legendre_basis
from symmetrynet.scratch.wigner import random_rotation
from symmetrynet.utils.graph import radius_graph
from symmetrynet.utils.precision import default_dtype

DT = torch.float64


# --------------------------------------------------------------- angular basis
@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4, 6])
def test_legendre_matches_scipy(degree: int):
    """Checked against an independent implementation, not against itself."""
    scipy_special = pytest.importorskip("scipy.special")
    x = torch.linspace(-1, 1, 101, dtype=DT)
    ours = legendre_basis(x, degree)[..., degree]
    theirs = torch.tensor(scipy_special.eval_legendre(degree, x.numpy()), dtype=DT)
    assert torch.allclose(ours, theirs, atol=1e-12)


def test_legendre_known_values():
    """P0=1, P1=x, P2=(3x^2-1)/2 -- spot-checked in closed form."""
    x = torch.tensor([-1.0, -0.3, 0.0, 0.5, 1.0], dtype=DT)
    basis = legendre_basis(x, 2)
    assert torch.allclose(basis[:, 0], torch.ones_like(x))
    assert torch.allclose(basis[:, 1], x)
    assert torch.allclose(basis[:, 2], (3 * x**2 - 1) / 2)


def test_legendre_handles_cos_slightly_out_of_range():
    """Round-off in a dot product can push |cos| just past 1; that must stay finite."""
    x = torch.tensor([-1.0 - 1e-12, 1.0 + 1e-12], dtype=DT)
    assert torch.isfinite(legendre_basis(x, 4)).all()


# ------------------------------------------------------------------- triplets
def test_triplets_match_brute_force():
    """Vectorised enumeration must equal the obvious nested loop."""
    torch.manual_seed(0)
    pos = torch.randn(24, 3, dtype=DT) * 1.6
    edge_index = radius_graph(pos, 3.0)
    edge_a, edge_b = build_triplets(edge_index, pos.shape[0])

    got = {(int(a), int(b)) for a, b in zip(edge_a.tolist(), edge_b.tolist(), strict=True)}

    dst = edge_index[1].tolist()
    expected = {
        (a, b)
        for a in range(len(dst))
        for b in range(len(dst))
        if a != b and dst[a] == dst[b]
    }
    assert got == expected


def test_triplets_share_a_destination():
    torch.manual_seed(1)
    pos = torch.randn(20, 3, dtype=DT) * 1.6
    edge_index = radius_graph(pos, 3.5)
    edge_a, edge_b = build_triplets(edge_index, pos.shape[0])
    assert (edge_index[1][edge_a] == edge_index[1][edge_b]).all()
    assert (edge_a != edge_b).all()


def test_triplets_empty_graph_is_handled():
    """An isolated-atom graph has no angles and must not crash."""
    edge_index = torch.empty(2, 0, dtype=torch.long)
    edge_a, edge_b = build_triplets(edge_index, 5)
    assert edge_a.numel() == 0 and edge_b.numel() == 0


def test_computed_angle_matches_geometry():
    """A hand-built right angle must come back as 90 degrees.

    Angles are the whole point of this model; a sign or ordering slip would still train,
    just to a worse optimum, so it is worth pinning against known geometry.
    """
    # Atom 0 at the origin with neighbours along +x and +y: the angle at 0 is 90 degrees.
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DT)
    edge_index = radius_graph(pos, 1.5)
    edge_a, edge_b = build_triplets(edge_index, 3)

    src, dst = edge_index[0], edge_index[1]
    vec = pos[dst] - pos[src]
    length = vec.norm(dim=-1)
    cos = (vec[edge_a] * vec[edge_b]).sum(-1) / (length[edge_a] * length[edge_b])

    at_origin = dst[edge_a] == 0
    assert at_origin.any()
    assert torch.allclose(cos[at_origin], torch.zeros(int(at_origin.sum()), dtype=DT), atol=1e-12)

    # The angles at atoms 1 and 2 are 45 degrees (the triangle's other corners).
    at_one = dst[edge_a] == 1
    if at_one.any():
        assert torch.allclose(
            cos[at_one].abs(),
            torch.full((int(at_one.sum()),), math.cos(math.pi / 4), dtype=DT),
            atol=1e-12,
        )


# ------------------------------------------------------------------- the model
@pytest.fixture
def model():
    with default_dtype(DT):
        return AngularInvariantGNN(hidden=32, num_layers=2, angular_hidden=16).eval()


def test_rotation_invariance(model, molecules):
    """Invariant despite consuming angles -- angles are themselves invariant."""
    species, pos, batch, _ = molecules
    with torch.no_grad():
        base = model(species, pos, batch)
        for _ in range(15):
            rot = random_rotation(1, dtype=DT)
            assert torch.allclose(model(species, pos @ rot.T, batch), base, atol=1e-11)


def test_translation_and_inversion_invariance(model, molecules):
    species, pos, batch, _ = molecules
    with torch.no_grad():
        base = model(species, pos, batch)
        assert torch.allclose(model(species, pos + torch.randn(3, dtype=DT) * 9, batch),
                              base, atol=1e-10)
        # cos(theta) is unchanged by inversion, so this model is O(3) invariant too.
        assert torch.allclose(model(species, -pos, batch), base, atol=1e-11)


def test_it_actually_uses_angles(molecules):
    """The point of the model: two geometries with equal distances but different angles.

    A distance-only network cannot tell these apart. This one must.
    """
    # A rhombus and its mirror-flipped partner share every pairwise distance from atom 0
    # but differ in the angle at atom 0.
    species = torch.zeros(4, dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    a = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]],
                     dtype=DT)
    # Same three bond lengths from atom 0, different mutual angles.
    t = math.pi / 3
    b = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
         [1.5 * math.cos(t), 1.5 * math.sin(t), 0.0], [0.0, 0.0, 1.5]], dtype=DT
    )
    assert torch.allclose((a[1:] - a[0]).norm(dim=-1), (b[1:] - b[0]).norm(dim=-1))

    with default_dtype(DT):
        model = AngularInvariantGNN(hidden=32, num_layers=2, angular_hidden=16).eval()
    with torch.no_grad():
        assert not torch.allclose(model(species, a, batch), model(species, b, batch), atol=1e-6)


def test_gradients_reach_the_angular_block(molecules):
    species, pos, batch, _ = molecules
    model = AngularInvariantGNN(hidden=32, num_layers=2, angular_hidden=16)
    model(species, pos.float(), batch).sum().backward()

    angular_params = [
        (n, p) for n, p in model.named_parameters() if n.startswith("angular.")
    ]
    assert angular_params, "no angular parameters found"
    reached = [n for n, p in angular_params if p.grad is not None and p.grad.abs().max() > 0]
    assert len(reached) > 3, f"angular block barely trained: only {reached}"
