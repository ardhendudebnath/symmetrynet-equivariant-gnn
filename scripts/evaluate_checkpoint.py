"""Score a saved checkpoint on the test split and write its ``results.json``.

Training writes ``best.pt`` every time validation improves, but ``results.json`` only at
the very end. A run interrupted near the finish therefore leaves a perfectly good model
on disk with no recorded test score. Re-training to recover one number would be wasteful
and, worse, would not reproduce the same weights.

This script closes that gap: it rebuilds the model from the checkpoint's own config,
evaluates the test split, and writes the same ``results.json`` the training loop would
have. Optionally it recovers the per-epoch history by parsing a training log, so the
plots come out identical too.

Usage::

    python scripts/evaluate_checkpoint.py --run tfn_l2_f1_e200_s0 --log path/to/train.log
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from symmetrynet.data.qm9 import QM9DataModule  # noqa: E402
from symmetrynet.train import TrainConfig, build_model, evaluate  # noqa: E402

DEFAULT_RUNS = Path.home() / ".symmetrynet" / "runs"

# "epoch  12 | train  123.45 | val  120.00 meV | lr 1.0e-04 |  12.3s"
EPOCH_RE = re.compile(
    r"epoch\s+(\d+)\s*\|\s*train\s+([\d.]+)\s*\|\s*val\s+([\d.]+)\s*meV"
    r"\s*\|\s*lr\s+([\d.e+-]+)\s*\|\s*([\d.]+)s"
)


def parse_history(log_path: Path, run_name: str) -> list[dict]:
    """Pull one run's epoch lines out of a combined training log.

    A log may hold several runs back to back, so we only collect lines after the
    marker for the requested run and stop at the next marker.
    """
    history: list[dict] = []
    collecting = False
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("run           :"):
            collecting = line.split(":", 1)[1].strip() == run_name
            continue
        if not collecting:
            continue
        match = EPOCH_RE.search(line)
        if match:
            history.append(
                {
                    "epoch": int(match.group(1)),
                    "train_mae": float(match.group(2)) / 1000.0,
                    "val_mae": float(match.group(3)) / 1000.0,
                    "lr": float(match.group(4)),
                    "seconds": float(match.group(5)),
                }
            )
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run directory name")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--log", default=None, help="optional training log to recover history")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true", help="overwrite an existing results.json")
    args = parser.parse_args()

    run_dir = Path(args.runs_dir) / args.run
    checkpoint_path = run_dir / "best.pt"
    results_path = run_dir / "results.json"
    if not checkpoint_path.exists():
        raise SystemExit(f"no checkpoint at {checkpoint_path}")
    if results_path.exists() and not args.force:
        raise SystemExit(f"{results_path} already exists; pass --force to overwrite")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**checkpoint["config"])
    print(f"run        : {args.run}")
    print(f"model      : {cfg.model} (l_max={cfg.l_max}, multiplicity={cfg.multiplicity})")
    print(f"checkpoint : epoch {checkpoint['epoch']}, val {checkpoint['val_mae'] * 1000:.2f} meV")

    datamodule = QM9DataModule(
        target=cfg.target,
        root=cfg.data_root,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        train_fraction=cfg.train_fraction,
    )
    datamodule.setup()
    # Recomputed the same deterministic way the training run did, so the model is
    # rebuilt with exactly the aggregation normalisation it was trained with.
    avg_neighbors = datamodule.average_num_neighbors(cfg.cutoff)

    model = build_model(cfg, avg_neighbors).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    val_mae = evaluate(model, datamodule.val_loader(), datamodule, args.device)
    test_mae = evaluate(model, datamodule.test_loader(), datamodule, args.device)
    print(f"val  MAE   : {val_mae * 1000:.2f} meV  (checkpoint recorded "
          f"{checkpoint['val_mae'] * 1000:.2f})")
    print(f"TEST MAE   : {test_mae * 1000:.2f} meV")

    history: list[dict] = []
    if args.log:
        history = parse_history(Path(args.log), args.run)
        print(f"history    : recovered {len(history)} epochs from {args.log}")

    result = {
        "run_name": args.run,
        "config": asdict(cfg),
        "num_params": num_params,
        "avg_num_neighbors": avg_neighbors,
        "splits": datamodule.split_sizes(),
        "best_epoch": checkpoint["epoch"],
        "val_mae": val_mae,
        "test_mae": test_mae,
        "val_mae_meV": val_mae * 1000,
        "test_mae_meV": test_mae * 1000,
        "units": datamodule.units,
        "total_seconds": sum(h["seconds"] for h in history),
        "history": history,
        # Flagged so nobody later mistakes a recovered run for a clean one.
        "recovered_from_checkpoint": True,
        "epochs_completed": (max(h["epoch"] for h in history) + 1) if history else None,
    }
    results_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
