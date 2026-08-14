"""Aggregate the multi-seed data-efficiency grid and test whether the trend is real.

The claim under test is counter to the usual story about equivariant networks. The
literature's headline is that built-in symmetry buys *sample efficiency*, so the
equivariant advantage should be largest when data is scarce. A single seed suggested the
opposite on QM9's HOMO-LUMO gap. This script decides whether that survives replication.

What it reports, and why each piece matters:

* **Per-seed ratios.** Within one seed both models see a byte-identical split, so
  ``baseline_MAE / equivariant_MAE`` is a paired comparison. Across seeds the split
  changes too, so agreement means the effect is not an artifact of one partition.
* **Mean +/- standard deviation** at each training fraction.
* **A sign test on the direction.** With three seeds, quoting a p-value from a t-test
  would be false precision. The honest question is simpler: does the ratio increase from
  the smallest to the largest training set in *every* seed? Unanimity across independent
  splits is the strongest statement three seeds can support.
* **Whether the confidence intervals overlap** between the smallest and largest
  fractions, which is what decides if the trend is distinguishable from noise at all.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
DEFAULT_RUNS = Path.home() / ".symmetrynet" / "runs"

RUN_RE = re.compile(
    r"^(?P<model>baseline|painn|tfn)(?:_l(?P<l_max>\d+))?"
    r"_f(?P<fraction>[\d.]+)_e(?P<epochs>\d+)_s(?P<seed>\d+)$"
)
#: A best epoch this close to the cap means the run was still improving when it ended.
CONVERGENCE_MARGIN = 0.08


def collect(runs_dir: Path) -> dict:
    """(model, fraction, seed) -> record, keeping the largest epoch budget per key."""
    best: dict[tuple[str, float, int], dict] = {}
    for path in sorted(runs_dir.iterdir()):
        match = RUN_RE.match(path.name)
        results_file = path / "results.json"
        if not match or not results_file.exists():
            continue
        try:
            payload = json.loads(results_file.read_text())
        except json.JSONDecodeError:
            continue
        key = (match.group("model"), float(match.group("fraction")), int(match.group("seed")))
        epochs = int(match.group("epochs"))
        if key not in best or epochs > best[key]["epochs"]:
            completed = payload.get("epochs_completed") or payload["config"]["epochs"]
            best[key] = {
                "epochs": epochs,
                "run_name": payload["run_name"],
                "test_mae_meV": payload["test_mae_meV"],
                "best_epoch": payload["best_epoch"],
                "converged": payload["best_epoch"] < completed * (1.0 - CONVERGENCE_MARGIN),
                "train_size": payload["splits"]["train"],
            }
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--equivariant", default="painn", choices=["painn", "tfn"])
    # Deliberately *not* multiseed.json: run_experiments.py's multiseed suite writes its
    # own run summary under that name, and the two files have different schemas. Sharing
    # a filename meant whichever ran last silently clobbered the other.
    parser.add_argument("--out", default=str(RESULTS / "multiseed_analysis.json"))
    args = parser.parse_args()

    records = collect(Path(args.runs_dir))
    fractions = sorted({f for (_, f, _) in records})
    seeds = sorted({s for (m, _, s) in records if m == args.equivariant})

    ratios: dict[float, dict[int, float]] = defaultdict(dict)
    rows = []
    unconverged = []

    for fraction in fractions:
        for seed in seeds:
            base = records.get(("baseline", fraction, seed))
            equi = records.get((args.equivariant, fraction, seed))
            if not base or not equi:
                continue
            ratios[fraction][seed] = base["test_mae_meV"] / equi["test_mae_meV"]
            rows.append(
                {
                    "train_fraction": fraction,
                    "train_size": base["train_size"],
                    "seed": seed,
                    "baseline_mae_meV": base["test_mae_meV"],
                    "equivariant_mae_meV": equi["test_mae_meV"],
                    "ratio": ratios[fraction][seed],
                }
            )
            for rec in (base, equi):
                if not rec["converged"]:
                    unconverged.append(rec["run_name"])

    if not rows:
        raise SystemExit("no paired baseline/equivariant runs found yet")

    print(f"equivariant model: {args.equivariant}")
    print(f"seeds present    : {seeds}\n")

    header = f"{'train size':>11s} " + "".join(f"{'s' + str(s):>9s}" for s in seeds)
    header += f"{'mean':>9s}{'std':>8s}{'n':>4s}"
    print(header)
    print("-" * len(header))

    summary = []
    for fraction in fractions:
        per_seed = ratios[fraction]
        if not per_seed:
            continue
        values = [per_seed[s] for s in seeds if s in per_seed]
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        size = next(r["train_size"] for r in rows if r["train_fraction"] == fraction)
        cells = "".join(f"{per_seed[s]:>9.3f}" if s in per_seed else f"{'-':>9s}" for s in seeds)
        print(f"{size:>11,d} {cells}{mean:>9.3f}{std:>8.3f}{len(values):>4d}")
        summary.append(
            {"train_fraction": fraction, "train_size": size, "ratios": per_seed,
             "mean": mean, "std": std, "n": len(values)}
        )

    # ---------------------------------------------------------------- direction test
    print()
    lo, hi = summary[0], summary[-1]
    shared = [s for s in seeds if s in lo["ratios"] and s in hi["ratios"]]
    increases = [s for s in shared if hi["ratios"][s] > lo["ratios"][s]]

    print(f"Direction, {lo['train_size']:,} -> {hi['train_size']:,} molecules:")
    for s in shared:
        delta = hi["ratios"][s] - lo["ratios"][s]
        arrow = "grows" if delta > 0 else "shrinks"
        print(f"  seed {s}: {lo['ratios'][s]:.3f} -> {hi['ratios'][s]:.3f}  ({arrow} {delta:+.3f})")

    print(f"\n  advantage grows with data in {len(increases)}/{len(shared)} seeds")
    if shared and len(increases) == len(shared):
        print("  -> unanimous. The usual data-efficiency claim is contradicted here:")
        print("     the equivariant advantage is SMALLEST when data is scarcest.")
    elif not increases:
        print("  -> unanimous in the opposite direction: the data-efficiency claim holds.")
    else:
        print("  -> seeds disagree; no directional claim is supported by this evidence.")

    # Separation relative to spread is what decides whether the trend beats noise.
    if lo["n"] > 1 and hi["n"] > 1:
        gap = hi["mean"] - lo["mean"]
        pooled = (lo["std"] ** 2 + hi["std"] ** 2) ** 0.5
        print(f"\n  gap {gap:+.3f}, pooled sd {pooled:.3f}"
              f"  ->  {'separated' if abs(gap) > 2 * pooled else 'NOT separated'} at ~2 sd")

    if unconverged:
        print("\nWARNING: runs that hit their cap while still improving:")
        for name in sorted(set(unconverged)):
            print(f"  {name}")

    Path(args.out).write_text(json.dumps(
        {"equivariant": args.equivariant, "seeds": seeds, "per_run": rows,
         "per_fraction": summary}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
