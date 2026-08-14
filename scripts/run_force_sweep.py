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

#: Epochs per size, giving each point ~25k gradient steps at batch size 16.
EPOCHS = {250: 1600, 500: 800, 1000: 400, 2000: 200}

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
            for model, hparams in MODEL_HPARAMS.items():
                configs.append(
                    ForceTrainConfig(
                        model=model,
                        molecule=args.molecule,
                        train_size=size,
                        epochs=EPOCHS[size],
                        batch_size=args.batch_size,
                        patience=args.patience,
                        seed=seed,
                        out_dir=args.out_dir,
                        run_name=f"{model}_{args.molecule}_n{size}_s{seed}",
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
    print(f"{'run':<34s} {'N':>6s} {'force MAE':>11s} {'min':>7s}")
    for r in results:
        print(f"{r['run_name']:<34s} {r['config']['train_size']:>6d} "
              f"{r['test_force_mae']:>11.4f} {r['total_seconds'] / 60:>7.1f}")


if __name__ == "__main__":
    main()
