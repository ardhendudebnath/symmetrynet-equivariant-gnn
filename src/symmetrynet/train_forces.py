"""Training loop for energy-and-force prediction on MD17.

Kept separate from :mod:`symmetrynet.train` rather than bolted on with flags. Three things
genuinely differ, and each would have made the scalar loop worse to read:

* the loss combines two terms on quantities with different shapes and units;
* the reported metric is force MAE, in kcal/mol/A rather than meV;
* every training step needs a double backward, since the force is itself a gradient.

The force weight is 1000:1 by default, following the MD17 convention used by SchNet,
DimeNet, PaiNN and NequIP. The ratio looks extreme but is not arbitrary: there are ``3N``
force components against a single energy per configuration, and forces are the quantity
these models are actually judged on.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .data.md17 import MD17DataModule
from .models import MODEL_REGISTRY, ForceModel
from .train import EMA, cosine_schedule
from .utils.seed import seed_everything

__all__ = ["ForceTrainConfig", "train_forces", "evaluate_forces"]


@dataclass
class ForceTrainConfig:
    model: str = "painn"
    molecule: str = "ethanol"
    train_size: int = 1000
    epochs: int = 300
    batch_size: int = 16
    lr: float = 5e-4
    weight_decay: float = 1e-8
    warmup_epochs: int = 1
    min_lr_factor: float = 0.01
    ema_decay: float = 0.999
    grad_clip: float = 10.0
    force_weight: float = 1000.0
    energy_weight: float = 1.0
    cutoff: float = 5.0
    num_layers: int = 3
    l_max: int = 2
    multiplicity: int = 64
    hidden: int = 128
    num_radial: int = 20
    seed: int = 0
    patience: int = 50
    # Validate every N epochs. At small training sizes an "epoch" is a handful of steps,
    # so validating every one means thousands of full passes over 1000 configurations --
    # each needing a backward pass, since forces are gradients. Evaluating less often
    # costs a little resolution on the curve and saves most of the wall time.
    # `patience` counts *validations*, not epochs, so it stays meaningful either way.
    val_every: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_root: str | None = None
    out_dir: str = "runs_md17"
    run_name: str | None = None


def build_force_model(cfg: ForceTrainConfig, avg_num_neighbors: float) -> ForceModel:
    if cfg.model not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {cfg.model!r}")
    cls = MODEL_REGISTRY[cfg.model]
    candidates = {
        "num_species": 5,
        "cutoff": cfg.cutoff,
        "num_layers": cfg.num_layers,
        "num_radial": cfg.num_radial,
        "avg_num_neighbors": avg_num_neighbors,
        "l_max": cfg.l_max,
        "multiplicity": cfg.multiplicity,
        "hidden": cfg.hidden,
        "hidden_multiplicity": cfg.multiplicity,
    }
    import inspect

    accepted = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in candidates.items() if k in accepted}
    return ForceModel(cls(**kwargs))


def evaluate_forces(
    model: ForceModel, loader, datamodule: MD17DataModule, device: str
) -> dict[str, float]:
    """Force and energy MAE in physical units (kcal/mol/A and kcal/mol)."""
    model.eval()
    std = datamodule.standardizer
    assert std is not None

    force_err = energy_err = 0.0
    n_components = n_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        # No torch.no_grad(): the forward pass needs autograd to produce forces at all.
        energy, forces = model(batch.species, batch.pos, batch.batch, create_graph=False)
        target_e = std.encode_energy(batch.energy.view(-1))
        target_f = std.encode_force(batch.force)

        force_err += (forces - target_f).abs().sum().item()
        energy_err += (energy - target_e).abs().sum().item()
        n_components += target_f.numel()
        n_graphs += target_e.numel()

    return {
        "force_mae": std.decode_force_error(force_err / max(n_components, 1)),
        "energy_mae": std.decode_energy_error(energy_err / max(n_graphs, 1)),
    }


def train_forces(cfg: ForceTrainConfig) -> dict:
    seed_everything(cfg.seed)
    device = cfg.device

    datamodule = MD17DataModule(
        molecule=cfg.molecule,
        root=cfg.data_root,
        train_size=cfg.train_size,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )
    datamodule.setup()
    sizes = datamodule.split_sizes()
    avg_neighbors = datamodule.average_num_neighbors(cfg.cutoff)

    model = build_force_model(cfg, avg_neighbors).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    run_name = cfg.run_name or f"{cfg.model}_{cfg.molecule}_n{cfg.train_size}_s{cfg.seed}"
    out_dir = Path(cfg.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    std = datamodule.standardizer
    assert std is not None
    print(f"run           : {run_name}")
    print(f"model         : {cfg.model}  ({num_params:,} parameters)")
    print(f"molecule      : {cfg.molecule}  (forces, kcal/mol/A)")
    print(f"splits        : {sizes}")
    print(f"avg neighbors : {avg_neighbors:.2f} (cutoff {cfg.cutoff} A)")
    print(f"scale         : energy_mean={std.energy_mean:.2f}  force_rms={std.scale:.4f}")
    print(f"loss          : {cfg.force_weight:g} x force + {cfg.energy_weight:g} x energy")

    train_loader = datamodule.train_loader()
    val_loader = datamodule.val_loader()
    test_loader = datamodule.test_loader()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_schedule(step, total_steps, warmup_steps, cfg.min_lr_factor),
    )
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = -1
    stalled = 0
    start = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        running_f = running_e = 0.0
        seen = 0
        epoch_start = time.time()

        for batch in train_loader:
            batch = batch.to(device)
            target_e = std.encode_energy(batch.energy.view(-1))
            target_f = std.encode_force(batch.force)

            optimizer.zero_grad(set_to_none=True)
            energy, forces = model(batch.species, batch.pos, batch.batch)
            loss_f = nn.functional.l1_loss(forces, target_f)
            loss_e = nn.functional.l1_loss(energy, target_e)
            (cfg.force_weight * loss_f + cfg.energy_weight * loss_e).backward()

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            running_f += loss_f.item() * target_e.numel()
            running_e += loss_e.item() * target_e.numel()
            seen += target_e.numel()

        train_force_mae = std.decode_force_error(running_f / max(seen, 1))

        # Always validate on the final epoch so the best checkpoint is never missed
        # simply because the run ended between validations.
        is_last = epoch == cfg.epochs - 1
        if not (is_last or epoch % cfg.val_every == 0):
            continue

        backup = ema.copy_to(model) if ema is not None else None
        metrics = evaluate_forces(model, val_loader, datamodule, device)
        if backup is not None:
            model.load_state_dict(backup)

        improved = metrics["force_mae"] < best_val
        if improved:
            best_val, best_epoch = metrics["force_mae"], epoch
            stalled = 0
            state = ema.shadow if ema is not None else model.state_dict()
            torch.save(
                {"config": asdict(cfg), "state_dict": state, "epoch": epoch,
                 "val_force_mae": best_val},
                out_dir / "best.pt",
            )
        else:
            stalled += 1

        history.append(
            {"epoch": epoch, "train_force_mae": train_force_mae,
             "val_force_mae": metrics["force_mae"], "val_energy_mae": metrics["energy_mae"],
             "lr": scheduler.get_last_lr()[0], "seconds": time.time() - epoch_start}
        )
        print(
            f"epoch {epoch:4d} | train F {train_force_mae:8.4f} | val F "
            f"{metrics['force_mae']:8.4f} | val E {metrics['energy_mae']:8.4f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - epoch_start:6.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )

        if cfg.patience and stalled >= cfg.patience:
            print(f"early stopping: no improvement for {cfg.patience} epochs")
            break

    checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate_forces(model, test_loader, datamodule, device)

    result = {
        "run_name": run_name,
        "config": asdict(cfg),
        "num_params": num_params,
        "avg_num_neighbors": avg_neighbors,
        "splits": sizes,
        "best_epoch": best_epoch,
        "val_force_mae": best_val,
        "test_force_mae": test_metrics["force_mae"],
        "test_energy_mae": test_metrics["energy_mae"],
        "units": {"force": "kcal/mol/A", "energy": "kcal/mol"},
        "total_seconds": time.time() - start,
        "history": history,
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"\nbest epoch {best_epoch} | val F {best_val:.4f} | "
          f"TEST force MAE {test_metrics['force_mae']:.4f} kcal/mol/A | "
          f"test energy MAE {test_metrics['energy_mae']:.4f} kcal/mol | "
          f"{(time.time() - start) / 60:.1f} min")
    return result


def parse_args(argv: list[str] | None = None) -> ForceTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = ForceTrainConfig()
    for name, value in asdict(defaults).items():
        if isinstance(value, bool):
            parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=value)
        elif value is None:
            parser.add_argument(f"--{name}", type=str, default=None)
        else:
            parser.add_argument(f"--{name}", type=type(value), default=value)
    return ForceTrainConfig(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    train_forces(parse_args())
