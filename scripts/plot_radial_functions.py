"""Phase 6 interpretability: what did the radial networks actually learn?

In a Tensor Field Network every learnable interaction is factored as

    (learned function of distance)  x  (fixed equivariant tensor product)

so the *entire* learned content of a convolution is a set of scalar functions of the
interatomic distance.  That is unusually interpretable for a deep model: rather than
staring at weight matrices, you can simply plot the filters and see the length scales
the network chose to care about.

Produces a panel per interaction layer showing a sample of learned radial filters, plus
the fixed Bessel basis and cutoff envelope they are built from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from symmetrynet.train import TrainConfig, build_model  # noqa: E402

RUNS = Path.home() / ".symmetrynet" / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=str, default="tfn_l2_f1_e50_s0")
    parser.add_argument("--num-filters", type=int, default=12)
    parser.add_argument("--out", type=str, default=str(REPO_ROOT / "results" / "radial.png"))
    args = parser.parse_args()

    checkpoint_path = RUNS / args.run / "best.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"no checkpoint at {checkpoint_path}; train a model first")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**checkpoint["config"])
    model = build_model(cfg, checkpoint.get("avg_num_neighbors", 15.64))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"loaded {args.run}: {cfg.model}, l_max={cfg.l_max}, "
          f"val {checkpoint['val_mae'] * 1000:.2f} meV @ epoch {checkpoint['epoch']}")

    layers = list(getattr(model, "layers", getattr(model, "interactions", [])))
    if not layers:
        raise SystemExit("model exposes no interaction layers to inspect")

    dist = torch.linspace(0.01, cfg.cutoff, 400)
    with torch.no_grad():
        basis = model.radial_basis(dist)
        envelope = model.envelope(dist)

    ncols = len(layers) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3.9), sharex=True)
    axes = np.atleast_1d(axes)

    # Panel 0: the fixed inputs every learned filter is built from.
    ax = axes[0]
    for k in range(basis.shape[1]):
        ax.plot(dist, basis[:, k] * envelope, lw=1.2, alpha=0.85)
    ax.plot(dist, envelope * float(basis.abs().max()), color="k", ls="--", lw=1.4,
            label="cutoff envelope")
    ax.set_title("fixed Bessel basis x envelope", fontsize=10)
    ax.set_xlabel("interatomic distance (Å)")
    ax.set_ylabel("value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    rng = np.random.default_rng(0)
    for idx, layer in enumerate(layers):
        ax = axes[idx + 1]
        radial = getattr(layer, "radial", None) or getattr(layer, "filter_net", None)
        with torch.no_grad():
            weights = radial(basis) * envelope.unsqueeze(-1)
        weights = weights.numpy()

        # Show the filters with the most variation -- flat ones are uninformative.
        ranked = np.argsort(-weights.std(axis=0))
        chosen = ranked[: args.num_filters]
        rng.shuffle(chosen)
        for j in chosen:
            ax.plot(dist, weights[:, j], lw=1.3, alpha=0.85)

        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_title(f"layer {idx + 1}: learned radial filters\n"
                     f"({weights.shape[1]} total, {len(chosen)} shown)", fontsize=10)
        ax.set_xlabel("interatomic distance (Å)")
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"Every learned interaction in the TFN is a scalar function of distance "
        f"({args.run})",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
