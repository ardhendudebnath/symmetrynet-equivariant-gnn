"""Turn the experiment JSON files into the figures used in the README and report.

Reads whatever is present under ``results/`` and skips the rest, so it can be run
mid-sweep to see how things are going.

Produces:
* ``training_curves.png``  -- validation MAE per epoch, baseline vs equivariant.
* ``data_efficiency.png``  -- test MAE against training-set size, log-log.
* ``ablation_lmax.png``    -- test MAE against ``l_max``.
* ``results_table.md``     -- the numbers, for pasting into the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
RUNS = Path.home() / ".symmetrynet" / "runs"

BASELINE_COLOR = "#2a9d5c"
TFN_COLOR = "#1b6ca8"
LABELS = {"baseline": "Invariant baseline (distances only)", "tfn": "Equivariant TFN"}
COLORS = {"baseline": BASELINE_COLOR, "tfn": TFN_COLOR}


def load(name: str) -> list[dict] | None:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        print(f"skip {name}: {path} not found")
        return None
    return json.loads(path.read_text())


def load_history(run_name: str) -> list[dict] | None:
    path = RUNS / run_name / "results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["history"]


# ------------------------------------------------------------------ training curves
def plot_training_curves() -> None:
    rows = load("comparison")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for row in rows:
        history = load_history(row["run_name"])
        if not history:
            continue
        model = row["model"]
        epochs = [h["epoch"] for h in history]
        val = [h["val_mae"] * 1000 for h in history]
        ax.plot(epochs, val, color=COLORS[model], lw=2.0,
                label=f"{LABELS[model]}  (best {row['test_mae_meV']:.1f} meV test)")
        best = int(np.argmin(val))
        ax.plot(epochs[best], val[best], "o", color=COLORS[model], ms=6)

    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MAE (meV)")
    ax.set_yscale("log")
    ax.set_title("HOMO-LUMO gap on QM9 — identical training loop, only the model differs")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    out = RESULTS / "training_curves.png"
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


# ------------------------------------------------------------------ data efficiency
def plot_data_efficiency() -> None:
    rows = load("data_efficiency")
    if not rows:
        return
    fig, (ax, ax_ratio) = plt.subplots(
        1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    by_model: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append((row["train_size"], row["test_mae_meV"]))

    for model, points in by_model.items():
        points.sort()
        sizes = [p[0] for p in points]
        maes = [p[1] for p in points]
        ax.plot(sizes, maes, "o-", color=COLORS[model], lw=2.0, ms=7, label=LABELS[model])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training molecules")
    ax.set_ylabel("test MAE (meV)")
    ax.set_title("Data efficiency")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which="both")

    # How much baseline data would be needed to match the equivariant model?
    if {"baseline", "tfn"} <= by_model.keys():
        base = dict(by_model["baseline"])
        tfn = dict(by_model["tfn"])
        shared = sorted(set(base) & set(tfn))
        ratios = [base[s] / tfn[s] for s in shared]
        ax_ratio.plot(shared, ratios, "o-", color="#6a4c93", lw=2.0, ms=7)
        ax_ratio.axhline(1.0, color="0.6", ls="--", lw=1.0)
        ax_ratio.set_xscale("log")
        ax_ratio.set_xlabel("training molecules")
        ax_ratio.set_ylabel("baseline MAE / equivariant MAE")
        ax_ratio.set_title("Relative error reduction\n(above 1.0 favours the equivariant model)")
        ax_ratio.grid(alpha=0.25, which="both")
        for size, ratio in zip(shared, ratios, strict=True):
            ax_ratio.annotate(f"{ratio:.2f}x", (size, ratio), textcoords="offset points",
                              xytext=(0, 8), ha="center", fontsize=8.5)

    fig.tight_layout()
    out = RESULTS / "data_efficiency.png"
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


# ------------------------------------------------------------------------- ablation
def plot_ablation() -> None:
    rows = load("ablation")
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["l_max"])
    labels = [f"$\\ell_{{max}}={r['l_max']}$" for r in rows]
    values = [r["test_mae_meV"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    shades = ["#a8c9e0", "#5f9bc4", "#1b6ca8"]
    bars = ax.bar(labels, values, color=shades[: len(values)], width=0.6)

    baseline_rows = load("comparison") or []
    for row in baseline_rows:
        if row["model"] == "baseline":
            ax.axhline(row["test_mae_meV"], color=BASELINE_COLOR, ls="--", lw=1.6,
                       label=f"distance-only baseline ({row['test_mae_meV']:.1f} meV)")
            ax.legend(fontsize=9)

    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("test MAE (meV)")
    ax.set_title("How much is angular information worth?\n"
                 r"$\ell_{max}=0$ is an equivariant model with no angular paths",
                 fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out = RESULTS / "ablation_lmax.png"
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------- table
def write_table() -> None:
    lines = ["# Results\n"]
    for suite, title in (
        ("comparison", "Primary comparison (full 110k training split)"),
        ("ablation", "Ablation: maximum spherical-harmonic degree"),
        ("data_efficiency", "Data efficiency"),
    ):
        rows = load(suite)
        if not rows:
            continue
        lines += [f"\n## {title}\n"]
        lines += ["| run | model | l_max | train size | params | epochs | best epoch "
                  "| val MAE (meV) | test MAE (meV) | minutes |"]
        lines += ["|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(
            rows, key=lambda r: (r["model"], r["train_fraction"], r["l_max"] or -1)
        ):
            l_max = "—" if r["l_max"] is None else r["l_max"]
            lines.append(
                f"| {r['run_name']} | {r['model']} | {l_max} | {r['train_size']:,} "
                f"| {r['num_params']:,} | {r['epochs']} | {r['best_epoch']} "
                f"| {r['val_mae_meV']:.2f} | **{r['test_mae_meV']:.2f}** | {r['minutes']:.1f} |"
            )
    out = RESULTS / "results_table.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    plot_training_curves()
    plot_data_efficiency()
    plot_ablation()
    write_table()


if __name__ == "__main__":
    main()
