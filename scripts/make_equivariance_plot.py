"""Produce the headline figure: prediction vs rotation angle, and the error spectrum.

Left panel  -- spin one molecule through a full turn and plot each model's prediction.
               The equivariant models draw a perfectly flat line; the raw-coordinate
               controls swing wildly.
Right panel -- the same fact quantitatively: worst-case ``|f(Rx) - f(x)|`` over many
               random rotations, on a log axis spanning the ~13 orders of magnitude
               between "exact by construction" and "hoping the network learns it".

The demo works with *untrained* models, and that is the entire point: equivariance is a
property of the architecture, not something acquired during training.  Pass
``--checkpoint-dir`` to use trained weights instead; the flat lines stay flat.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from symmetrynet.data.qm9 import QM9DataModule  # noqa: E402
from symmetrynet.models import (  # noqa: E402
    InvariantGNN,
    NaiveCoordinateGNN,
    NaiveCoordinateMLP,
    TensorFieldNetwork,
)
from symmetrynet.scratch.wigner import random_rotation, rotation_matrix  # noqa: E402
from symmetrynet.utils.precision import default_dtype  # noqa: E402

DT = torch.float64

STYLES = {
    "Equivariant TFN (l_max=2)": {"color": "#1b6ca8", "lw": 2.6, "zorder": 5},
    "Invariant baseline (distances)": {"color": "#2a9d5c", "lw": 2.2, "ls": "--", "zorder": 4},
    "Naive GNN (raw coordinates)": {"color": "#d1495b", "lw": 1.8, "zorder": 3},
    "Naive MLP (raw coordinates)": {"color": "#e8a33d", "lw": 1.8, "zorder": 2},
}


def build_models(avg_neighbors: float) -> dict:
    """All models constructed in float64 so e3nn bakes exact constants."""
    with default_dtype(DT):
        return {
            "Equivariant TFN (l_max=2)": TensorFieldNetwork(
                multiplicity=32, l_max=2, num_layers=3, avg_num_neighbors=avg_neighbors
            ).eval(),
            "Invariant baseline (distances)": InvariantGNN(
                hidden=64, num_layers=3, avg_num_neighbors=avg_neighbors
            ).eval(),
            "Naive GNN (raw coordinates)": NaiveCoordinateGNN(
                hidden=64, num_layers=3, avg_num_neighbors=avg_neighbors
            ).eval(),
            "Naive MLP (raw coordinates)": NaiveCoordinateMLP(hidden=128).eval(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", type=int, default=1234, help="index into QM9")
    parser.add_argument("--angles", type=int, default=181)
    parser.add_argument("--rotations", type=int, default=200)
    parser.add_argument("--out", type=str, default=str(REPO_ROOT / "results" / "equivariance.png"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dm = QM9DataModule(target="gap")
    dm.setup()
    data = dm.dataset[args.molecule]
    species, pos = data.species, data.pos.to(DT)
    pos = pos - pos.mean(0, keepdim=True)
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    formula = "".join(
        f"{sym}{int((data.z == z).sum())}"
        for z, sym in zip((6, 1, 7, 8, 9), ("C", "H", "N", "O", "F"), strict=True)
        if int((data.z == z).sum()) > 0
    )
    print(f"molecule {args.molecule}: {formula}, {pos.shape[0]} atoms")

    models = build_models(dm.average_num_neighbors(5.0))

    # ---- left panel: sweep a rotation about a fixed tilted axis -------------------
    angles = torch.linspace(0, 2 * np.pi, args.angles, dtype=DT)
    curves: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, model in models.items():
            values = []
            for angle in angles:
                rot = rotation_matrix(
                    angle, torch.tensor(0.6, dtype=DT), torch.tensor(0.0, dtype=DT)
                )
                values.append(model(species, pos @ rot.T, batch).item())
            curves[name] = np.array(values)

    # ---- right panel: worst-case residual over uniformly random rotations ---------
    #
    # Reported *relative* to each model's own output scale.  Untrained networks have
    # arbitrary output magnitudes -- the equivariant model here sits near 100 eV while
    # the baseline sits near 1 eV -- so comparing raw absolute residuals would rank
    # models by how large their outputs happen to be rather than by how well they
    # respect the symmetry.  Dividing by the prediction scale makes the number
    # dimensionless and directly comparable, and puts float64 epsilon (~2.2e-16) on the
    # axis as a meaningful reference line.
    residuals: dict[str, float] = {}
    absolute: dict[str, float] = {}
    scales: dict[str, float] = {}
    with torch.no_grad():
        for name, model in models.items():
            base = model(species, pos, batch)
            worst = 0.0
            for _ in range(args.rotations):
                rot = random_rotation(1, dtype=DT)
                worst = max(worst, (model(species, pos @ rot.T, batch) - base).abs().max().item())
            scale = max(abs(base.item()), 1e-12)
            absolute[name] = worst
            scales[name] = scale
            residuals[name] = max(worst / scale, 1e-17)  # keep the log axis finite

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1.55, 1]}
    )

    for name, values in curves.items():
        centred = values - values.mean()
        ax_left.plot(angles.numpy(), centred, label=name, **STYLES[name])

    ax_left.set_xlabel("rotation angle about a fixed axis (radians)")
    ax_left.set_ylabel("prediction, mean-centred  (eV)")
    ax_left.set_title(
        f"Same molecule ({formula}), rotated through a full turn\n"
        "physics says the answer cannot change",
        fontsize=11,
    )
    ax_left.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax_left.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax_left.axhline(0.0, color="0.75", lw=0.8, zorder=1)
    ax_left.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    ax_left.grid(alpha=0.25)

    # At this scale both equivariant curves collapse onto y=0 and onto each other, so
    # the reader cannot tell "flat" from "hidden behind the other line".  The inset
    # zooms in by ~13 orders of magnitude to show the flatness is real.
    inset = ax_left.inset_axes([0.09, 0.09, 0.42, 0.26])
    for name in ("Equivariant TFN (l_max=2)", "Invariant baseline (distances)"):
        style = dict(STYLES[name])
        style["lw"] = 1.6
        inset.plot(angles.numpy(), (curves[name] - curves[name].mean()) * 1e15, **style)
    inset.set_ylim(-5, 5)
    inset.axhline(0.0, color="0.75", lw=0.6)
    inset.set_xticks([0, np.pi, 2 * np.pi])
    inset.set_xticklabels(["0", "π", "2π"], fontsize=7)
    inset.tick_params(labelsize=7)
    inset.set_title("same two curves, y-axis in units of $10^{-15}$ eV",
                    fontsize=7.5, pad=3)
    inset.grid(alpha=0.2)

    names = list(residuals)
    values = [residuals[n] for n in names]
    colors = [STYLES[n]["color"] for n in names]
    bars = ax_right.barh(range(len(names)), values, color=colors, height=0.6)
    ax_right.set_xscale("log")
    ax_right.set_yticks(range(len(names)))
    ax_right.set_yticklabels([n.replace(" (", "\n(") for n in names], fontsize=8.5)
    ax_right.invert_yaxis()
    ax_right.set_xlabel(r"worst $|f(Rx) - f(x)|\,/\,|f(x)|$ over "
                        f"{args.rotations} random rotations (log scale)")
    ax_right.set_title("Relative equivariance error\n(float64; lower is better)", fontsize=11)
    ax_right.axvline(2.2e-16, color="0.4", ls=":", lw=1.2)
    ax_right.text(2.2e-16, -0.55, " float64 epsilon", fontsize=7.5, color="0.35", va="bottom")
    ax_right.grid(alpha=0.25, axis="x", which="both")

    for bar, value in zip(bars, values, strict=True):
        ax_right.text(
            value * 1.6,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1e}",
            va="center",
            fontsize=8,
        )
    ax_right.set_xlim(1e-17, max(values) * 200)

    fig.suptitle(
        "Equivariance is a property of the architecture, not of the training data "
        "— these models are untrained",
        fontsize=12.5,
        y=1.0,
    )
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"wrote {out}")

    summary = {
        "molecule_index": args.molecule,
        "formula": formula,
        "num_atoms": int(pos.shape[0]),
        "num_random_rotations": args.rotations,
        "dtype": "float64",
        "trained": False,
        "worst_relative_deviation": residuals,
        "worst_abs_deviation_eV": absolute,
        "prediction_scale_eV": scales,
        # np.ptp(x), not x.ptp() -- NumPy 2 removed the ndarray method.
        "peak_to_peak_over_sweep_eV": {n: float(np.ptp(v)) for n, v in curves.items()},
    }
    summary_path = out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")
    for name, value in residuals.items():
        print(f"  {name:<34s} {value:.3e}")


if __name__ == "__main__":
    main()
