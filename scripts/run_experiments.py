"""Run the experiment suites that produce the project's headline numbers.

Three suites, all sharing one training loop so nothing is tuned per-model:

``comparison``
    Baseline vs equivariant model on the full 110k training split.  The primary result.
``data_efficiency``
    Both models at 10 / 25 / 50 / 100% of the training set.  Epochs are scaled up for
    smaller fractions so every point gets a comparable number of gradient steps --
    otherwise the curve would measure "how many updates did it get" rather than
    "how much data did it need".  Early stopping keeps that from wasting time.
``ablation``
    ``l_max`` in {0, 1, 2}, isolating how much the angular representation is worth.
    ``l_max=0`` is the informative control: an equivariant architecture stripped of all
    angular information, which should land near the distance-only baseline.

Usage::

    python scripts/run_experiments.py comparison --epochs 50
    python scripts/run_experiments.py data_efficiency
    python scripts/run_experiments.py ablation
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from symmetrynet.train import TrainConfig, train  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_RUN_DIR = Path.home() / ".symmetrynet" / "runs"


def canonical_run_name(cfg: TrainConfig) -> str:
    """A name determined by the config, so identical configs share one run directory.

    The suites overlap: the ablation's ``l_max=2`` point and the data-efficiency 100%
    points are bit-for-bit the same runs as the primary comparison (same model, same
    hyperparameters, same seed).  Naming by config rather than by suite means each is
    trained once and reused everywhere -- exactly equivalent, not an approximation, and
    it cuts several hours off a full sweep.

    It also makes the sweep resumable: an interrupted run picks up where it left off
    instead of starting the whole suite again.
    """
    parts = [cfg.model]
    if cfg.model == "tfn":
        parts.append(f"l{cfg.l_max}")
        if not cfg.batch_norm:
            parts.append("nobn")
    parts.append(f"f{cfg.train_fraction:g}")
    parts.append(f"e{cfg.epochs}")
    parts.append(f"s{cfg.seed}")
    return "_".join(parts)


def _record(suite: str, results: list[dict], tag: str | None = None) -> list[Path]:
    """Persist a compact summary; the full history stays in each run directory.

    Written twice: once to ``<suite>.json`` (what the plotting script reads) and once to
    ``<suite>_<tag>.json``, keyed by the epoch budget.  Re-running a suite at a different
    budget would otherwise silently overwrite the previous summary, destroying exactly
    the comparison the re-run was meant to enable.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{suite}.json"
    summary = [
        {
            "run_name": r["run_name"],
            "model": r["config"]["model"],
            # Only the equivariant model has a meaningful l_max; recording the config
            # default for the others would read as "the baseline uses l_max=2".
            "l_max": r["config"]["l_max"] if r["config"]["model"] == "tfn" else None,
            "train_fraction": r["config"]["train_fraction"],
            "seed": r["config"]["seed"],
            "epochs": r["config"]["epochs"],
            "num_params": r["num_params"],
            "train_size": r["splits"]["train"],
            "best_epoch": r["best_epoch"],
            "val_mae_meV": r["val_mae_meV"],
            "test_mae_meV": r["test_mae_meV"],
            "minutes": r["total_seconds"] / 60,
        }
        for r in results
    ]
    blob = json.dumps(summary, indent=2)
    path.write_text(blob)
    written = [path]
    if tag:
        archival = RESULTS_DIR / f"{suite}_{tag}.json"
        archival.write_text(blob)
        written.append(archival)
    return written


def _existing(cfg: TrainConfig) -> dict | None:
    """Return a completed result for this exact config, if one is already on disk."""
    path = Path(cfg.out_dir) / cfg.run_name / "results.json"
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None  # interrupted mid-write; retrain
    # Only reuse if the run actually finished the epoch budget it was asked for.
    if result.get("config", {}).get("epochs") != cfg.epochs:
        return None
    return result


def _run_all(
    suite: str,
    configs: list[TrainConfig],
    *,
    skip_existing: bool = True,
    tag: str | None = None,
) -> None:
    results: list[dict] = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n{'=' * 78}\n[{i}/{len(configs)}] {cfg.run_name}\n{'=' * 78}", flush=True)
        if skip_existing:
            cached = _existing(cfg)
            if cached is not None:
                print(f"reusing completed run ({cached['test_mae_meV']:.2f} meV test)",
                      flush=True)
                results.append(cached)
                _record(suite, results, tag)
                continue
        try:
            results.append(train(cfg))
        except Exception:  # noqa: BLE001 - one failed config must not lose the rest
            print(f"FAILED: {cfg.run_name}", flush=True)
            traceback.print_exc()
            continue
        # Write after every run so partial results survive an interruption.
        written = _record(suite, results, tag)
        print("summary -> " + ", ".join(str(p) for p in written), flush=True)

    print(f"\n{'=' * 78}\n{suite} complete: {len(results)}/{len(configs)} runs\n{'=' * 78}")
    for r in results:
        print(f"  {r['run_name']:<40s} test {r['test_mae_meV']:8.2f} meV")


def _base(args, **overrides) -> TrainConfig:
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        multiplicity=args.multiplicity,
        hidden=args.hidden,
        num_layers=args.num_layers,
        seed=args.seed,
        out_dir=args.out_dir,
        patience=args.patience,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.run_name = canonical_run_name(cfg)
    return cfg


def suite_comparison(args) -> list[TrainConfig]:
    return [
        _base(args, model="baseline"),
        _base(args, model="tfn", l_max=2),
    ]


def suite_data_efficiency(args) -> list[TrainConfig]:
    configs = []
    for fraction in (0.1, 0.25, 0.5, 1.0):
        # Comparable gradient-step budget across fractions, capped so the 10% point
        # does not run forever; early stopping usually ends it well before the cap.
        epochs = min(int(round(args.epochs / fraction)), args.max_epochs)
        for model, extra in (("baseline", {}), ("tfn", {"l_max": 2})):
            configs.append(
                _base(args, model=model, train_fraction=fraction, epochs=epochs, **extra)
            )
    return configs


def suite_ablation(args) -> list[TrainConfig]:
    return [_base(args, model="tfn", l_max=ell) for ell in (0, 1, 2)]


#: Epoch budget per training fraction, chosen empirically so every point *converges*
#: rather than hitting its cap while still improving.  These are not proportional to the
#: data size: a 10% split has a tenth as many gradient steps per epoch, so it needs many
#: more epochs, while its smaller training set also converges in fewer total steps.
#: Verified against seed 0 -- best epochs came out 135-189 (of 250), 109-145 (of 200),
#: 178-312 (of 400) and 111-135 (of 200), all comfortably inside budget.
CONVERGED_EPOCHS = {0.1: 250, 0.25: 200, 0.5: 400, 1.0: 200}

#: Per-model hyperparameters, pinned to exactly what seed 0 ran.  Replication is
#: worthless if the replicas differ from the original in some quiet default.
MODEL_HPARAMS = {
    "baseline": {"hidden": 128, "num_layers": 4, "num_radial": 8},
    "painn": {"hidden": 128, "num_layers": 3, "num_radial": 20},
}


def suite_multiseed(args) -> list[TrainConfig]:
    """The data-efficiency grid repeated over several seeds, for error bars.

    The finding under test -- that the equivariant advantage *grows* with dataset size,
    contradicting the usual data-efficiency claim -- currently rests on a single seed. A
    contrarian result without error bars is not a result.

    Note that ``seed`` drives both the train/val/test split and the weight
    initialisation. That is deliberate: within one seed the two models see byte-identical
    data, so the per-seed *ratio* is a clean paired comparison, while across seeds the
    split changes, so a trend that survives is robust to the particular split rather than
    an artifact of one lucky partition.
    """
    configs = []
    for seed in args.seeds:
        for fraction, epochs in CONVERGED_EPOCHS.items():
            for model, hparams in MODEL_HPARAMS.items():
                cfg = _base(
                    args,
                    model=model,
                    train_fraction=fraction,
                    epochs=epochs,
                    seed=seed,
                    **hparams,
                )
                cfg.run_name = canonical_run_name(cfg)
                configs.append(cfg)
    return configs


SUITES = {
    "comparison": suite_comparison,
    "data_efficiency": suite_data_efficiency,
    "ablation": suite_ablation,
    "multiseed": suite_multiseed,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=sorted(SUITES))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max_epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--multiplicity", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="seeds for the multiseed suite (each drives both the split and the init)",
    )
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_RUN_DIR))
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="retrain even if a completed run with the same config already exists",
    )
    args = parser.parse_args()

    # The multiseed suite sets its epoch budget per fraction and ignores --epochs, so
    # tagging it "e50" would label the file with a number none of its runs used.
    tag = f"s{''.join(str(s) for s in args.seeds)}" if args.suite == "multiseed" \
        else f"e{args.epochs}"
    _run_all(
        args.suite,
        SUITES[args.suite](args),
        skip_existing=not args.retrain,
        tag=tag,
    )


if __name__ == "__main__":
    main()
