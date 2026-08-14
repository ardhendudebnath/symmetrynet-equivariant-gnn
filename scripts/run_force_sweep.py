"""Data-efficiency sweep on MD17 forces -- the tensorial counterpart to the QM9 study.

The question
------------
On QM9's HOMO-LUMO gap (a *scalar*) the equivariant advantage grew with dataset size: the
benefit showed up as a steeper scaling exponent, not as sample efficiency. The standard
data-efficiency claim, however, was made for **forces**, which are tensorial (l=1).

If forces show the opposite signature -- advantage largest when data is scarce -- then the
contradiction becomes a boundary condition, and the useful statement is
"equivariance buys sample efficiency for tensorial targets and scaling for scalar ones".
If forces show the *same* signature, the claim is weaker than usually presented.

Either outcome is worth having, which is the point of running it.

Budget
------
Epochs are scaled inversely with training size so every point receives roughly the same
number of gradient steps (~25k). With a fixed epoch count the smallest set would get a
sixteenth of the updates of the largest, which is the confound that spoiled the first
version of the QM9 curve. Early stopping ends runs that converge sooner.

Usage::

    python scripts/run_force_sweep.py --seeds 0
    python scripts/run_force_sweep.py --seeds 0 1 2      # after the direction is known
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from symmetrynet.train_forces import ForceTrainConfig, train_forces  # noqa: E402

RESULTS = REPO_ROOT / "results"
DEFAULT_RUNS = Path.home() / ".symmetrynet" / "runs_md17"

#: Training sizes, in MD17's conventional small-data regime (papers quote N=1000).
TRAIN_SIZES = (250, 500, 1000, 2000)

#: Gradient steps per run, which is the budget that actually matters.
#:
#: The first version of this sweep fixed *epochs* per size to equalise steps at ~25k, and
#: every single run then finished with its best epoch at or next to the last -- i.e. none
#: of them converged, and the comparison was meaningless. The tail improvements looked
#: small only because the cosine schedule had annealed the learning rate to zero, the same
#: trap that made the TFN look 24% worse than it was on QM9.
#:
#: 200k steps is ~8x that budget, chosen from measured throughput (~17 ms/step, so ~1 h per
#: run). Runs that genuinely converge stop earlier via patience; runs that do not are
#: flagged rather than quietly reported.
TARGET_STEPS = 200_000

#: Fraction of the budget a run may stall for before early stopping. Expressed as a
#: fraction rather than a fixed epoch count because an "epoch" at N=250 is 16 steps and at
#: N=2000 is 125 -- a flat patience would be eight times stricter at the small end.
PATIENCE_FRACTION = 0.15


def epochs_for(train_size: int, batch_size: int) -> int:
    """Epochs needed to reach ``TARGET_STEPS`` gradient updates at this training size."""
    steps_per_epoch = max(1, -(-train_size // batch_size))  # ceil division
    return max(1, TARGET_STEPS // steps_per_epoch)

#: Pinned per-model hyperparameters, matching the QM9 experiments so the two studies are
#: comparable rather than merely adjacent.
MODEL_HPARAMS = {
    "baseline": {"hidden": 128, "num_layers": 4, "num_radial": 8},
    "painn": {"hidden": 128, "num_layers": 3, "num_radial": 20},
}


def build_configs(args) -> list[ForceTrainConfig]:
    configs = []
    for seed in args.seeds:
        for size in TRAIN_SIZES:
            epochs = epochs_for(size, args.batch_size)
            for model, hparams in MODEL_HPARAMS.items():
                configs.append(
                    ForceTrainConfig(
                        model=model,
                        molecule=args.molecule,
                        train_size=size,
                        epochs=epochs,
                        batch_size=args.batch_size,
                        patience=int(epochs * PATIENCE_FRACTION),
                        seed=seed,
                        out_dir=args.out_dir,
                        # Budget is in the name: a run trained to a different step count
                        # is a different experiment and must not silently be reused.
                        run_name=f"{model}_{args.molecule}_n{size}_e{epochs}_s{seed}",
                        **hparams,
                    )
                )
    return configs


def existing(cfg: ForceTrainConfig) -> dict | None:
    path = Path(cfg.out_dir) / (cfg.run_name or "") / "results.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if payload.get("config", {}).get("epochs") == cfg.epochs else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", default="ethanol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--out_dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    configs = build_configs(args)
    results: list[dict] = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n{'=' * 78}\n[{i}/{len(configs)}] {cfg.run_name}"
              f"  ({cfg.epochs} epochs)\n{'=' * 78}", flush=True)
        cached = None if args.retrain else existing(cfg)
        if cached is not None:
            print(f"reusing completed run "
                  f"(test force MAE {cached['test_force_mae']:.4f})", flush=True)
            results.append(cached)
        else:
            try:
                results.append(train_forces(cfg))
            except Exception:  # noqa: BLE001 - one failure must not lose the rest
                print(f"FAILED: {cfg.run_name}", flush=True)
                traceback.print_exc()
                continue

        RESULTS.mkdir(parents=True, exist_ok=True)
        summary = [
            {
                "run_name": r["run_name"],
                "model": r["config"]["model"],
                "molecule": r["config"]["molecule"],
                "train_size": r["config"]["train_size"],
                "seed": r["config"]["seed"],
                "epochs": r["config"]["epochs"],
                "best_epoch": r["best_epoch"],
                "num_params": r["num_params"],
                "test_force_mae": r["test_force_mae"],
                "test_energy_mae": r["test_energy_mae"],
                "minutes": r["total_seconds"] / 60,
            }
            for r in results
        ]
        (RESULTS / "md17_forces.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 78}\nforce sweep complete: {len(results)}/{len(configs)}\n{'=' * 78}")
    print(f"{'run':<38s} {'N':>6s} {'force MAE':>11s} {'converged':>10s} {'min':>7s}")

    # This flags whether a run finished its schedule. It does NOT prove the schedule was
    # long enough, and the distinction cost a sweep to learn.
    #
    # The obvious test -- "best epoch landed near the end, so it was still improving" --
    # false-positives on flat curves, where the argmin sits anywhere in the plateau by
    # noise alone: a validated run here was at 94% of its schedule having improved 0.05%
    # over its final tenth. Measuring the improvement *rate* fixes that.
    #
    # But neither test detects an insufficient budget. A short cosine schedule anneals the
    # learning rate to zero, so the curve flattens whether or not the model has reached its
    # potential. Measured directly: a 25k-step run showed a reassuring 0.39% tail gain and
    # was still beaten by 20% at 200k steps. Budget sufficiency can only be established by
    # comparing budgets, which is why TARGET_STEPS was validated that way rather than by
    # trusting a flat tail.
    stale = []
    for r in results:
        history = r.get("history") or []
        converged, tail_gain = True, 0.0
        if len(history) >= 10:
            curve = [h["val_force_mae"] for h in history]
            reference = curve[int(len(curve) * 0.9)]
            best = min(curve)
            tail_gain = (reference - best) / reference if reference > 0 else 0.0
            converged = tail_gain < 0.005  # still falling by <0.5% over the final tenth
        r["_tail_gain"] = tail_gain
        if not converged:
            stale.append(f"{r['run_name']} (still improving {tail_gain * 100:.1f}%)")
        flag = "NO" if any(r["run_name"] in s for s in stale) else "yes"
        print(f"{r['run_name']:<38s} {r['config']['train_size']:>6d} "
              f"{r['test_force_mae']:>11.4f} {flag:>10s} "
              f"{r['total_seconds'] / 60:>7.1f}")

    if stale:
        print("\nWARNING: these runs were still improving when the budget expired, so")
        print("their numbers understate the model and must not be quoted as converged:")
        for name in stale:
            print(f"  {name}")
        print("\nRaise TARGET_STEPS and re-run before drawing any conclusion.")


if __name__ == "__main__":
    main()
