"""Fast head-to-head probe on a training subset, to choose a fair configuration.

Motivation
----------
The first full-budget comparison had the equivariant model *behind* the distance-only
baseline. Before reporting that, it is worth separating two very different causes:

1. **Unequal scalar capacity.** The comparison gave the baseline ``hidden=128``, i.e. 128
   scalar channels per layer, while the TFN's ``multiplicity=64`` spends its width across
   ``64x0e + 64x1o + 64x2e``. The equivariant model therefore had *half* the scalar
   capacity, and the HOMO-LUMO gap is a scalar.
2. **Slower optimisation.** Tensor-product models are known to converge more slowly per
   epoch; a 50-epoch budget may simply favour the simpler model.

Both are legitimate findings, but they call for opposite responses -- fix the setup, or
report the budget honestly. This script runs short matched trials on a data subset so the
question is settled with measurements rather than intuition.

Usage::

    python scripts/probe_configs.py --fraction 0.1 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from symmetrynet.train import TrainConfig, train  # noqa: E402

RESULTS = REPO_ROOT / "results"
RUNS = Path.home() / ".symmetrynet" / "runs_probe"


def build_configs(args) -> list[TrainConfig]:
    common = {
        "target": "gap",
        "epochs": args.epochs,
        "train_fraction": args.fraction,
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "seed": args.seed,
        "out_dir": str(RUNS),
    }
    configs = [
        TrainConfig(
            model="baseline", hidden=128, lr=5e-4,
            run_name="probe_baseline_h128_lr5e-4", **common,
        )
    ]
    # TFN: vary width and learning rate, the two prime suspects.
    for multiplicity in args.multiplicities:
        for lr in args.lrs:
            configs.append(
                TrainConfig(
                    model="tfn",
                    l_max=2,
                    multiplicity=multiplicity,
                    lr=lr,
                    run_name=f"probe_tfn_m{multiplicity}_lr{lr:g}",
                    **common,
                )
            )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lrs", type=float, nargs="+", default=[5e-4, 1e-3])
    parser.add_argument("--multiplicities", type=int, nargs="+", default=[64, 128])
    args = parser.parse_args()

    results = []
    configs = build_configs(args)
    for i, cfg in enumerate(configs, 1):
        print(f"\n{'=' * 78}\n[{i}/{len(configs)}] {cfg.run_name}\n{'=' * 78}", flush=True)
        results.append(train(cfg))
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "probe.json").write_text(
            json.dumps(
                [
                    {
                        "run_name": r["run_name"],
                        "model": r["config"]["model"],
                        "multiplicity": r["config"]["multiplicity"],
                        "hidden": r["config"]["hidden"],
                        "lr": r["config"]["lr"],
                        "num_params": r["num_params"],
                        "test_mae_meV": r["test_mae_meV"],
                        "val_mae_meV": r["val_mae_meV"],
                        "minutes": r["total_seconds"] / 60,
                    }
                    for r in results
                ],
                indent=2,
            )
        )

    print(f"\n{'=' * 78}\nprobe summary (train fraction {args.fraction}, "
          f"{args.epochs} epochs)\n{'=' * 78}")
    for r in sorted(results, key=lambda r: r["test_mae_meV"]):
        print(f"  {r['run_name']:<34s} {r['num_params']:>9,d} params  "
              f"test {r['test_mae_meV']:8.2f} meV  ({r['total_seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
