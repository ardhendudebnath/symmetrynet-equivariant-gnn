"""Radial bases, cutoffs, graph construction, and batch independence.

None of this is about symmetry, but each item here caused (or could plausibly cause) a
silent accuracy loss rather than a crash, which is exactly the kind of bug a test suite
earns its keep on.
"""

from __future__ import annotations

import math

import pytest
import torch

from symmetrynet.models import InvariantGNN, TensorFieldNetwork
from symmetrynet.nn.radial import (
    BesselBasis,
    CosineCutoff,
    GaussianSmearing,
    PolynomialCutoff,
)
from symmetrynet.utils.graph import radius_graph, scatter_mean, scatter_sum
from symmetrynet.utils.precision import default_dtype

DT = torch.float64
CUTOFF = 5.0


# ------------------------------------------------------------------- radial basis
def test_bessel_basis_is_unit_scale_when_normalized():
    """The conditioning fix: basis values must be order 1, not order 0.3.

    A basis of the wrong scale does not fail loudly.  It propagates into the radial
    MLP, whose initialisation assumes unit-variance inputs, and ends up shrinking the
    tensor-product weights that e3nn normalises assuming unit variance -- attenuating
    the whole message pathway.  See README finding (3).
    """
    basis = BesselBasis(num_basis=8, cutoff=CUTOFF, normalize=True)
    # Sample distances the way pair separations actually distribute (~r^2 dr).
    u = torch.rand(20000, dtype=torch.float32)
    dist = CUTOFF * u.pow(1 / 3)
    values = basis(dist)
    rms = values.pow(2).mean().sqrt().item()
    assert 0.5 < rms < 2.0, f"expected order-1 RMS, got {rms}"


def test_bessel_normalization_is_opt_out():
    raw = BesselBasis(num_basis=8, cutoff=CUTOFF, normalize=False)
    normed = BesselBasis(num_basis=8, cutoff=CUTOFF, normalize=True)
    assert raw.scale == 1.0
    assert normed.scale > 1.0
    dist = torch.rand(64) * CUTOFF
    # Normalisation is a pure rescale; it must not change the basis shape.
    assert torch.allclose(normed(dist), raw(dist) * normed.scale, atol=1e-6)


def test_bessel_basis_finite_at_zero_distance():
    """A 0/0 here would poison a whole batch with NaN."""
    assert torch.isfinite(BesselBasis(8, CUTOFF)(torch.zeros(4))).all()


@pytest.mark.parametrize("cutoff_cls", [PolynomialCutoff, CosineCutoff])
def test_cutoff_vanishes_at_the_boundary(cutoff_cls):
    """A discontinuity at the cutoff makes the energy jump as atoms cross it."""
    envelope = cutoff_cls(CUTOFF)
    assert envelope(torch.tensor([CUTOFF])).abs().item() < 1e-12
    assert envelope(torch.tensor([CUTOFF + 1.0])).abs().item() == 0.0
    assert envelope(torch.tensor([0.0])).item() == pytest.approx(1.0, abs=1e-6)


def test_polynomial_cutoff_derivative_vanishes_quadratically():
    """The derivative must vanish to *second* order at the cutoff.

    Asserting ``|f'(c - eps)| < tol`` for one fixed ``eps`` would be a weak (and
    tolerance-dependent) check -- the value there is legitimately O(eps^2), not zero.
    The meaningful statement is the rate: shrinking ``eps`` tenfold must shrink the
    derivative roughly a hundredfold.  That is what makes forces continuous as an atom
    crosses the cutoff, rather than merely small.
    """
    envelope = PolynomialCutoff(CUTOFF)

    def slope(eps: float) -> float:
        d = torch.tensor([CUTOFF - eps], dtype=DT, requires_grad=True)
        envelope(d).backward()
        return abs(d.grad.item())

    coarse, fine = slope(1e-3), slope(1e-4)
    assert coarse > 0.0
    assert 50 < coarse / fine < 200, f"expected ~100x, got {coarse / fine:.1f}"


def test_gaussian_smearing_shape():
    out = GaussianSmearing(num_basis=16, cutoff=CUTOFF)(torch.rand(32) * CUTOFF)
    assert out.shape == (32, 16)
    assert (out >= 0).all() and (out <= 1).all()


# ----------------------------------------------------------------- graph building
def test_radius_graph_respects_cutoff_and_excludes_self_loops():
    pos = torch.randn(60, 3, dtype=DT) * 2.5
    edge_index = radius_graph(pos, CUTOFF)
    src, dst = edge_index
    assert (src != dst).all(), "self-loops must not appear"
    assert ((pos[dst] - pos[src]).norm(dim=-1) <= CUTOFF + 1e-9).all()


def test_radius_graph_never_crosses_graphs():
    pos = torch.randn(40, 3, dtype=DT)
    batch = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
    src, dst = radius_graph(pos, float("inf"), batch)
    assert (batch[src] == batch[dst]).all()


def test_radius_graph_matches_brute_force():
    pos = torch.randn(50, 3, dtype=DT) * 2.0
    got = radius_graph(pos, 3.0, chunk_size=7)  # small chunks to exercise the loop
    dist = torch.cdist(pos, pos)
    expected = ((dist <= 3.0) & ~torch.eye(50, dtype=torch.bool)).sum().item()
    assert got.shape[1] == expected


def test_radius_graph_max_neighbors_caps_degree():
    pos = torch.randn(40, 3, dtype=DT) * 0.5  # dense: everything within the cutoff
    _, dst = radius_graph(pos, float("inf"), max_num_neighbors=5)
    assert torch.bincount(dst, minlength=40).max().item() <= 5


def test_scatter_ops():
    src = torch.tensor([[1.0], [2.0], [4.0]])
    index = torch.tensor([0, 0, 1])
    assert torch.allclose(scatter_sum(src, index, 2), torch.tensor([[3.0], [4.0]]))
    assert torch.allclose(scatter_mean(src, index, 2), torch.tensor([[1.5], [4.0]]))


def test_scatter_mean_handles_empty_bins():
    """An empty bin must give 0, not a division by zero."""
    out = scatter_mean(torch.ones(2, 1), torch.tensor([0, 0]), 3)
    assert torch.isfinite(out).all() and out[2].item() == 0.0


# ------------------------------------------------------- batch-composition safety
@pytest.mark.parametrize("model_name", ["tfn", "baseline"])
def test_prediction_is_independent_of_batch_composition(model_name):
    """A molecule's prediction must not depend on what it was batched with.

    Worth asserting specifically because the equivariant model now uses ``BatchNorm``.
    In eval mode it uses running statistics, so predictions stay per-molecule -- but
    that is a property to verify, not assume, since getting it wrong would make test
    metrics depend on batch size.
    """
    torch.manual_seed(0)
    with default_dtype(DT):
        model = (
            TensorFieldNetwork(multiplicity=16, l_max=2, num_layers=2)
            if model_name == "tfn"
            else InvariantGNN(hidden=32, num_layers=2)
        ).eval()

    sizes = [9, 13, 11]
    species = [torch.randint(0, 5, (n,)) for n in sizes]
    pos = [torch.randn(n, 3, dtype=DT) * 1.6 for n in sizes]

    with torch.no_grad():
        alone = model(species[0], pos[0], torch.zeros(sizes[0], dtype=torch.long))
        together = model(
            torch.cat(species),
            torch.cat(pos),
            torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(sizes)]),
        )
    assert torch.allclose(alone[0], together[0], atol=1e-10)


def test_batch_norm_flag_changes_the_model():
    """The `--no-batch_norm` configuration must actually differ, so the README's
    before/after conditioning result stays reproducible."""
    with default_dtype(DT):
        with_bn = TensorFieldNetwork(multiplicity=16, l_max=2, num_layers=2, batch_norm=True)
        without = TensorFieldNetwork(multiplicity=16, l_max=2, num_layers=2, batch_norm=False)
    assert with_bn.layers[0].norm is not None
    assert without.layers[0].norm is None


def test_batch_norm_controls_activation_growth():
    """Regression guard for the conditioning bug: activations must not compound."""
    torch.manual_seed(0)
    species = torch.randint(0, 5, (48,))
    pos = torch.randn(48, 3) * 1.6
    batch = torch.arange(4).repeat_interleave(12)

    stds: dict[bool, list[float]] = {}
    for flag in (False, True):
        torch.manual_seed(0)
        model = TensorFieldNetwork(
            multiplicity=32, l_max=2, num_layers=4, avg_num_neighbors=15.64, batch_norm=flag
        ).train()
        captured: list[float] = []
        handles = [
            layer.register_forward_hook(
                # `sink=captured` binds the list per iteration rather than closing over
                # the loop variable, so the two passes cannot write into each other.
                lambda mod, inp, out, sink=captured: sink.append(float(out.detach().std()))
            )
            for layer in model.layers
        ]
        model(species, pos, batch)
        for handle in handles:
            handle.remove()
        stds[flag] = captured

    # Without normalisation the signal grows several-fold layer over layer.
    assert stds[False][-1] / stds[False][0] > 3.0
    # With it, every layer sits near unit scale.
    assert all(0.3 < s < 3.0 for s in stds[True]), stds[True]


def test_avg_num_neighbors_is_plausible():
    """A wrong aggregation constant silently rescales every layer."""
    pos = torch.randn(500, 3, dtype=DT) * 3.0
    _, dst = radius_graph(pos, CUTOFF)
    degree = torch.bincount(dst, minlength=500).float().mean().item()
    assert degree > 1.0
    assert math.isfinite(degree)
