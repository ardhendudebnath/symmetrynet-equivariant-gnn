r"""Fit neural-scaling exponents to the data-efficiency runs.

Why this is worth doing on top of the ratio table
-------------------------------------------------
The multi-seed sweep established that the equivariant advantage *grows* with dataset size.
That is descriptive. The same data supports a sharper, quantitative claim.

Empirical error curves usually follow a power law in the training set size,

.. math::  \mathrm{MAE}(N) \;\approx\; A \, N^{-b},

where :math:`b` is the **scaling exponent** -- how fast error falls as data is added -- and
:math:`A` is an offset. Taking the ratio of two such curves,

.. math::  \frac{\mathrm{MAE}_{\text{base}}(N)}{\mathrm{MAE}_{\text{equi}}(N)}
           \;=\; \frac{A_{\text{base}}}{A_{\text{equi}}}\; N^{\,b_{\text{equi}} - b_{\text{base}}},

so the ratio can only *increase* with :math:`N` if
:math:`b_{\text{equi}} > b_{\text{base}}`. The observed trend therefore implies the
equivariant model has a **steeper exponent**, not merely a better constant. That is a much
more useful statement: an offset advantage is a fixed discount, whereas an exponent
advantage compounds with every additional molecule.

This script fits both exponents per seed, reports them with spread, and checks whether the
difference is unanimous across seeds. It also reports :math:`R^2`, because an exponent is
meaningless if the points are not actually on a power law -- with four points that check is
not optional.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"

BASELINE_COLOR = "#2a9d5c"
PAINN_COLOR = "#8e44ad"


def fit_power_law(sizes: list[int], maes: list[float]) -> tuple[float, float, float]:
    """Least squares on log-log axes.

    Returns ``(exponent b, prefactor A, r_squared)`` for ``MAE = A * N**(-b)``.
    """
    log_n = np.log(np.asarray(sizes, dtype=float))
    log_mae = np.log(np.asarray(maes, dtype=float))
    slope, intercept = np.polyfit(log_n, log_mae, 1)

    predicted = slope * log_n + intercept
    ss_res = float(((log_mae - predicted) ** 2).sum())
    ss_tot = float(((log_mae - log_mae.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return -float(slope), float(np.exp(intercept)), r_squared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(RESULTS / "multiseed_analysis.json"))
    parser.add_argument("--out-json", default=str(RESULTS / "scaling.json"))
    parser.add_argument("--out-png", default=str(RESULTS / "scaling.png"))
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    per_run = payload["per_run"]
    seeds = sorted({row["seed"] for row in per_run})

    # (model, seed) -> parallel lists of sizes and MAEs
    curves: dict[tuple[str, int], tuple[list[int], list[float]]] = {}
    for model, key in (("baseline", "baseline_mae_meV"), ("equivariant", "equivariant_mae_meV")):
        for seed in seeds:
            rows = sorted(
                (r for r in per_run if r["seed"] == seed), key=lambda r: r["train_size"]
            )
            if len(rows) < 3:
                continue  # a power law through two points is not a fit
            curves[(model, seed)] = (
                [r["train_size"] for r in rows],
                [r[key] for r in rows],
            )

    if not curves:
        raise SystemExit("not enough points per seed to fit a power law")

    fits: dict[str, dict[int, dict]] = {"baseline": {}, "equivariant": {}}
    print(f"MAE(N) = A * N^-b     fitted over {len(seeds)} seeds\n")
    print(f"{'model':<13s} {'seed':>5s} {'exponent b':>12s} {'R^2':>8s}")
    print("-" * 42)
    for model in ("baseline", "equivariant"):
        for seed in seeds:
            if (model, seed) not in curves:
                continue
            sizes, maes = curves[(model, seed)]
            b, a, r2 = fit_power_law(sizes, maes)
            fits[model][seed] = {"exponent": b, "prefactor": a, "r_squared": r2,
                                 "sizes": sizes, "maes": maes}
            print(f"{model:<13s} {seed:>5d} {b:>12.4f} {r2:>8.4f}")

    summary = {}
    print()
    for model in ("baseline", "equivariant"):
        values = [f["exponent"] for f in fits[model].values()]
        r2s = [f["r_squared"] for f in fits[model].values()]
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[model] = {"mean": mean, "std": std, "min_r_squared": min(r2s)}
        print(f"  {model:<13s} b = {mean:.4f} +/- {std:.4f}   (min R^2 = {min(r2s):.4f})")

    # ------------------------------------------------- does the exponent differ?
    shared = [s for s in seeds if s in fits["baseline"] and s in fits["equivariant"]]
    deltas = {s: fits["equivariant"][s]["exponent"] - fits["baseline"][s]["exponent"]
              for s in shared}
    delta_mean = statistics.fmean(deltas.values())
    delta_std = statistics.stdev(deltas.values()) if len(deltas) > 1 else 0.0

    print("\nExponent difference (equivariant - baseline), per seed:")
    for seed in shared:
        sign = "steeper" if deltas[seed] > 0 else "shallower"
        print(f"  seed {seed}: {deltas[seed]:+.4f}   ({sign})")
    print(f"\n  mean {delta_mean:+.4f} +/- {delta_std:.4f}")

    steeper = sum(1 for d in deltas.values() if d > 0)
    print(f"  equivariant exponent is steeper in {steeper}/{len(deltas)} seeds")
    if steeper == len(deltas) and delta_mean > 2 * max(delta_std, 1e-12):
        print("  -> the equivariant model scales BETTER with data, not merely better")
        print("     by a constant factor. An offset advantage is a fixed discount; an")
        print("     exponent advantage compounds with every molecule added.")
    elif steeper == len(deltas):
        print("  -> direction unanimous, but the gap is within ~2 sd of the spread:")
        print("     suggestive of a steeper exponent, not established by this evidence.")
    else:
        print("  -> seeds disagree; no exponent claim is supported.")

    # ------------------------------------------------------------------- figure
    fig, (ax, ax_delta) = plt.subplots(1, 2, figsize=(12.6, 4.8),
                                       gridspec_kw={"width_ratios": [1.4, 1]})
    grid = None
    for model, colour, label in (
        ("baseline", BASELINE_COLOR, "Invariant baseline (distances only)"),
        ("equivariant", PAINN_COLOR, "Equivariant PaiNN"),
    ):
        for i, seed in enumerate(sorted(fits[model])):
            f = fits[model][seed]
            ax.plot(f["sizes"], f["maes"], "o", color=colour, ms=6, alpha=0.75,
                    label=label if i == 0 else None)
            grid = np.linspace(min(f["sizes"]) * 0.9, max(f["sizes"]) * 1.1, 50)
            ax.plot(grid, f["prefactor"] * grid ** (-f["exponent"]),
                    color=colour, lw=1.2, alpha=0.5)
        b = summary[model]["mean"]
        s = summary[model]["std"]
        ax.plot([], [], color=colour, lw=2.4, label=f"   fit: $b$ = {b:.3f} $\\pm$ {s:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training molecules $N$")
    ax.set_ylabel("test MAE (meV)")
    ax.set_title("Error scaling: $\\mathrm{MAE} \\propto N^{-b}$\n"
                 "steeper slope = error falls faster as data is added", fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, which="both")

    ax_delta.axhline(0.0, color="0.55", ls="--", lw=1.2)
    ax_delta.bar([str(s) for s in shared], [deltas[s] for s in shared],
                 color=PAINN_COLOR, width=0.55)
    ax_delta.axhline(delta_mean, color="#333", ls=":", lw=1.4,
                     label=f"mean {delta_mean:+.3f}")
    ax_delta.set_xlabel("seed")
    ax_delta.set_ylabel("$b_{\\mathrm{equivariant}} - b_{\\mathrm{baseline}}$")
    ax_delta.set_title("Exponent difference per seed\n"
                       "above zero = equivariance improves the scaling exponent",
                       fontsize=11)
    ax_delta.legend(fontsize=9)
    ax_delta.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=170)
    print(f"\nwrote {args.out_png}")

    Path(args.out_json).write_text(json.dumps({
        "model": "MAE(N) = A * N**-b",
        "seeds": seeds,
        "per_model": summary,
        "per_seed_exponents": {
            m: {str(s): f["exponent"] for s, f in fits[m].items()} for m in fits
        },
        "per_seed_r_squared": {
            m: {str(s): f["r_squared"] for s, f in fits[m].items()} for m in fits
        },
        "exponent_difference": {
            "per_seed": {str(s): d for s, d in deltas.items()},
            "mean": delta_mean,
            "std": delta_std,
            "steeper_in_n_seeds": steeper,
            "n_seeds": len(deltas),
        },
    }, indent=2))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
