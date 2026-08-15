"""Consolidate the MD17 force runs, selecting the training budget on validation.

Why budget selection, rather than a convergence heuristic
---------------------------------------------------------
Three heuristics were tried for "did this run converge", and each failed on a real case:

1. *Best epoch sits near the end.* False-positives on flat curves: once validation error
   plateaus, the argmin lands anywhere in the plateau by noise. A converged run was
   flagged at 94% of its schedule having improved 0.05% over its final tenth.
2. *Validation error still falling over the last 10%.* Fixed that, but passed a genuinely
   under-trained run at 0.39% -- a short cosine schedule anneals the learning rate to zero,
   so the curve flattens whether or not the model reached its potential.
3. *The same test on longer runs.* Reported 35% "still improving" for a run whose
   validation error was **rising**. On an overfitted curve the minimum occurs early, so
   comparing the 90% mark against the minimum measures overfitting, not progress.

The third case is the informative one. At N=250 the distance-only baseline reaches its
best validation score at ~25k steps and then degrades: by 600k its validation error has
risen from 0.57 to 0.83. More training makes it worse. No single budget is right for every
model and dataset size, so there is nothing for a convergence test to detect.

The standard answer is simply to treat the budget as a hyperparameter: train at several
budgets, pick per configuration using **validation**, and report the corresponding test
score. Selecting on validation rather than test is what keeps this honest -- picking the
budget with the best test score would be tuning on the test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
DEFAULT_RUNS = Path.home() / ".symmetrynet" / "runs_md17"

TRAIN_SIZES = (250, 500, 1000, 2000)


def gradient_steps(config: dict) -> int:
    per_epoch = max(1, -(-config["train_size"] // config["batch_size"]))
    return config["epochs"] * per_epoch


def collect(runs_dir: Path, molecule: str, seed: int) -> dict:
    """(model, train_size) -> the run with the best *validation* force MAE."""
    best: dict[tuple[str, int], dict] = {}
    for path in sorted(runs_dir.iterdir()):
        results_file = path / "results.json"
        if not results_file.exists():
            continue
        try:
            payload = json.loads(results_file.read_text())
        except json.JSONDecodeError:
            continue
        cfg = payload["config"]
        if cfg["molecule"] != molecule or cfg["seed"] != seed:
            continue

        key = (cfg["model"], cfg["train_size"])
        record = {
            "run_name": payload["run_name"],
            "steps": gradient_steps(cfg),
            "val_force_mae": payload["val_force_mae"],
            "test_force_mae": payload["test_force_mae"],
            "test_energy_mae": payload["test_energy_mae"],
            "best_epoch": payload["best_epoch"],
            "epochs": cfg["epochs"],
        }
        if key not in best or record["val_force_mae"] < best[key]["val_force_mae"]:
            best[key] = record
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--molecule", default="ethanol")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(RESULTS / "md17_forces.json"))
    args = parser.parse_args()

    best = collect(Path(args.runs_dir), args.molecule, args.seed)
    if not best:
        raise SystemExit("no MD17 runs found")

    print(f"molecule {args.molecule}, seed {args.seed}")
    print("budget chosen per configuration on validation; test force MAE reported\n")
    header = (f"{'N':>6s} | {'baseline':>9s} {'steps':>7s} | {'PaiNN':>9s} {'steps':>7s} "
              f"| {'ratio':>6s}")
    print(header)
    print("-" * len(header))

    rows = []
    for size in TRAIN_SIZES:
        b = best.get(("baseline", size))
        p = best.get(("painn", size))
        if not b or not p:
            continue
        ratio = b["test_force_mae"] / p["test_force_mae"]
        rows.append(
            {"train_size": size, "baseline": b, "painn": p, "ratio": ratio}
        )
        print(f"{size:>6d} | {b['test_force_mae']:>9.4f} {b['steps'] // 1000:>6d}k "
              f"| {p['test_force_mae']:>9.4f} {p['steps'] // 1000:>6d}k | {ratio:>6.2f}")

    if len(rows) > 1:
        first, last = rows[0]["ratio"], rows[-1]["ratio"]
        direction = "GROWS" if last > first else "shrinks"
        print(f"\nratio {first:.2f} -> {last:.2f}: equivariant advantage {direction} "
              f"with more data")
        print("(ratio > 1 means the equivariant model is better)")
        print("\nComparison with the QM9 scalar-target study:")
        if last > first:
            print("  same direction. The advantage grows with data on a tensorial target")
            print("  too, so the usual data-efficiency claim is not rescued by switching")
            print("  from a scalar target to forces.")
        else:
            print("  opposite direction. Equivariance buys sample efficiency for")
            print("  tensorial targets and scaling for scalar ones -- the contradiction")
            print("  found on QM9 is a boundary, not a general failure.")

    Path(args.out).write_text(json.dumps(
        {"molecule": args.molecule, "seed": args.seed,
         "budget_selected_on": "validation", "points": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
