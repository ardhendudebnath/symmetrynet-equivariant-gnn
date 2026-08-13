"""Rebuild the data-efficiency summary from whichever runs are most converged.

The learning-curve points were not all trained under one epoch budget, and they should
not have been. A point at 10% data sees a fifth as many gradient steps per epoch as one
at 50%, so a fixed epoch count trains the small-data points to convergence while leaving
the large-data points short -- which is exactly the artifact that made an earlier version
of this curve misleading.

The right protocol for a learning curve is to train every point *to convergence* and let
the epoch budget differ. This script assembles the summary accordingly: for each training
fraction it finds every completed run and keeps the one with the largest epoch budget,
then reports whether that run actually converged (best epoch comfortably inside the
budget) rather than simply running out of schedule.

Run it after adding a longer run for any fraction; it supersedes the older, shorter one.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
DEFAULT_RUNS = Path.home() / ".symmetrynet" / "runs"

# e.g. tfn_l2_f0.5_e400_s0  /  baseline_f1_e200_s0  /  painn_f0.25_e200_s0
RUN_RE = re.compile(
    r"^(?P<model>baseline|tfn|painn)(?:_l(?P<l_max>\d+))?(?:_(?P<flags>nobn))?"
    r"_f(?P<fraction>[\d.]+)_e(?P<epochs>\d+)_s(?P<seed>\d+)$"
)

#: A run counts as converged if its best epoch left this much of the schedule unused.
#: A best epoch at the very end means the model was still improving when the budget ran
#: out, so the number understates it.
CONVERGENCE_MARGIN = 0.08


def collect(runs_dir: Path, seed: int) -> dict[tuple[str, float], dict]:
    """Best available run per (model, fraction), preferring the largest epoch budget."""
    best: dict[tuple[str, float], dict] = {}
    for path in sorted(runs_dir.iterdir()):
        match = RUN_RE.match(path.name)
        results_file = path / "results.json"
        if not match or not results_file.exists():
            continue
        if match.group("flags") or int(match.group("seed")) != seed:
            continue  # skip the deliberately mis-conditioned no-batchnorm run
        try:
            payload = json.loads(results_file.read_text())
        except json.JSONDecodeError:
            print(f"  skipping {path.name}: unreadable results.json")
            continue

        key = (match.group("model"), float(match.group("fraction")))
        epochs = int(match.group("epochs"))
        if key not in best or epochs > best[key]["epochs"]:
            best[key] = {"epochs": epochs, "name": path.name, "payload": payload}
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(RESULTS / "data_efficiency.json"))
    args = parser.parse_args()

    best = collect(Path(args.runs_dir), args.seed)
    if not best:
        raise SystemExit("no completed runs found")

    rows = []
    for (model, fraction), entry in sorted(best.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        r = entry["payload"]
        epochs = r["config"]["epochs"]
        completed = r.get("epochs_completed") or epochs
        best_epoch = r["best_epoch"]
        converged = best_epoch < completed * (1.0 - CONVERGENCE_MARGIN)
        rows.append(
            {
                "run_name": r["run_name"],
                "model": model,
                "l_max": r["config"]["l_max"] if model == "tfn" else None,
                "train_fraction": fraction,
                "seed": args.seed,
                "epochs": epochs,
                "epochs_completed": completed,
                "num_params": r["num_params"],
                "train_size": r["splits"]["train"],
                "best_epoch": best_epoch,
                "converged": converged,
                "val_mae_meV": r["val_mae_meV"],
                "test_mae_meV": r["test_mae_meV"],
                "minutes": r["total_seconds"] / 60,
                "recovered_from_checkpoint": r.get("recovered_from_checkpoint", False),
            }
        )

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}\n")

    header = (
        f"{'fraction':>9s} {'model':<10s} {'best/total':>12s} "
        f"{'converged':>10s} {'test MAE':>10s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['train_fraction']:>9.2f} {row['model']:<10s} "
            f"{row['best_epoch']:>5d}/{row['epochs_completed']:<6d} "
            f"{('yes' if row['converged'] else 'NO'):>10s} "
            f"{row['test_mae_meV']:>9.2f}"
        )

    stale = [r for r in rows if not r["converged"]]
    if stale:
        print("\nWARNING: these points hit their epoch cap while still improving and "
              "understate the model:")
        for row in stale:
            print(f"  {row['run_name']} (best epoch {row['best_epoch']} of "
                  f"{row['epochs_completed']})")


if __name__ == "__main__":
    main()
