"""Training loop shared by every model in the comparison.

One loop, one optimiser schedule, one early-stopping rule.  If the baseline and the
equivariant model were tuned separately the headline number would be meaningless, so
the only thing that varies between runs is ``--model``.

Reported metric is **MAE in physical units** (meV for the HOMO-LUMO gap), which is what
the QM9 literature quotes.  Training happens in standardised units; the conversion back
multiplies by the target's standard deviation and adds no offset.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from .data.qm9 import QM9DataModule
from .models import MODEL_REGISTRY
from .utils.seed import seed_everything

__all__ = ["TrainConfig", "train", "evaluate", "build_model"]


@dataclass
class TrainConfig:
    model: str = "tfn"
    target: str = "gap"
    epochs: int = 60
    batch_size: int = 96
    lr: float = 5e-4
    weight_decay: float = 1e-8
    warmup_epochs: int = 1
    min_lr_factor: float = 0.01
    ema_decay: float = 0.999
    grad_clip: float = 10.0
    cutoff: float = 5.0
    num_layers: int = 4
    l_max: int = 2
    multiplicity: int = 64
    hidden: int = 128
    num_radial: int = 8
    # AngularInvariantGNN only. Kept modest because the triplet tensor is the memory
    # bottleneck of the whole project: ~414k triplets per batch of 96 molecules, each
    # carrying autograd intermediates through four angular blocks.
    angular_hidden: int = 16
    # Legendre degree. 2 matches the TFN's l_max=2: by the addition theorem, P_l up to
    # l=2 is exactly the invariant content of spherical harmonics up to l=2, so the two
    # models receive comparable angular information.
    max_degree: int = 2
    train_fraction: float = 1.0
    seed: int = 0
    num_workers: int = 0
    amp: bool = False
    # Equivariant BatchNorm on the TFN's residual stream.  Off reproduces the
    # badly-conditioned configuration described in the README's finding (3), where
    # activations compound ~4x per layer and the model underfits despite being exactly
    # equivariant.  Kept as a flag so that result stays reproducible.
    batch_norm: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_root: str | None = None
    out_dir: str = "runs"
    run_name: str | None = None
    patience: int = 0  # 0 disables early stopping
    extra: dict = field(default_factory=dict)


class EMA:
    """Exponential moving average of parameters, with warmup-adjusted decay.

    QM9 validation curves are noisy enough that the final-epoch weights are a poor
    estimate of a model's real quality; averaging typically buys several meV for free.

    The warmup matters more than it looks.  A fixed decay of 0.999 has an effective
    horizon of ~1000 steps, so a run with only a few hundred steps -- exactly what the
    10% data-efficiency point produces -- would be evaluated on weights that are still
    mostly the random initialisation.  That does not merely add noise, it makes small
    training fractions look catastrophically bad and would have manufactured a
    beautiful, entirely fake data-efficiency curve.  Ramping the decay in as
    ``(1+t)/(10+t)`` keeps the average responsive early and converges to ``decay`` later.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.step = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        decay = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if value.dtype.is_floating_point:
                shadow.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                shadow.copy_(value)

    def copy_to(self, model: nn.Module) -> dict[str, Tensor]:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return backup


def build_model(cfg: TrainConfig, avg_num_neighbors: float) -> nn.Module:
    """Instantiate a model, passing only the arguments it actually accepts."""
    if cfg.model not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {cfg.model!r}; choose from {sorted(MODEL_REGISTRY)}")
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
        "batch_norm": cfg.batch_norm,
        "angular_hidden": cfg.angular_hidden,
        "max_degree": cfg.max_degree,
    }
    import inspect

    accepted = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in candidates.items() if k in accepted}
    return cls(**kwargs)


def cosine_schedule(step: int, total: int, warmup: int, min_factor: float) -> float:
    """Linear warmup then cosine decay, as a multiplier on the base learning rate."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: nn.Module, loader, datamodule: QM9DataModule, device: str) -> float:
    """Mean absolute error in physical units."""
    model.eval()
    std = datamodule.standardizer
    assert std is not None
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.species, batch.pos, batch.batch)
        target = std.encode(datamodule.extract_target(batch))
        total += (pred - target).abs().sum().item()
        count += target.numel()
    return std.decode_error(total / max(count, 1))


def train(cfg: TrainConfig) -> dict:
    seed_everything(cfg.seed)
    device = cfg.device

    datamodule = QM9DataModule(
        target=cfg.target,
        root=cfg.data_root,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        train_fraction=cfg.train_fraction,
        num_workers=cfg.num_workers,
    )
    datamodule.setup()
    sizes = datamodule.split_sizes()
    avg_neighbors = datamodule.average_num_neighbors(cfg.cutoff)

    model = build_model(cfg, avg_neighbors).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    run_name = cfg.run_name or f"{cfg.model}_{cfg.target}_f{cfg.train_fraction}_s{cfg.seed}"
    out_dir = Path(cfg.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"run           : {run_name}")
    print(f"device        : {device}")
    print(f"model         : {cfg.model}  ({num_params:,} parameters)")
    print(f"target        : {datamodule.target_name} [{datamodule.units}]")
    print(f"splits        : {sizes}")
    print(f"avg neighbors : {avg_neighbors:.2f} (cutoff {cfg.cutoff} A)")
    print(f"standardizer  : mean={datamodule.standardizer.mean:.4f} "
          f"std={datamodule.standardizer.std:.4f}")

    train_loader = datamodule.train_loader()
    val_loader = datamodule.val_loader()
    test_loader = datamodule.test_loader()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_schedule(step, total_steps, warmup_steps, cfg.min_lr_factor),
    )
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    scaler = torch.amp.GradScaler(device, enabled=cfg.amp and device == "cuda")
    std = datamodule.standardizer

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        running, seen = 0.0, 0
        epoch_start = time.time()

        for batch in train_loader:
            batch = batch.to(device)
            target = std.encode(datamodule.extract_target(batch))

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=scaler.is_enabled()):
                pred = model(batch.species, batch.pos, batch.batch)
                # L1 matches the reported metric; L2 would over-weight the tail.
                loss = torch.nn.functional.l1_loss(pred, target)

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            running += loss.item() * target.numel()
            seen += target.numel()

        train_mae = std.decode_error(running / max(seen, 1))

        # Validate with the averaged weights, since those are what we would ship.
        backup = ema.copy_to(model) if ema is not None else None
        val_mae = evaluate(model, val_loader, datamodule, device)
        if backup is not None:
            model.load_state_dict(backup)

        improved = val_mae < best_val
        if improved:
            best_val, best_epoch = val_mae, epoch
            epochs_without_improvement = 0
            state = ema.shadow if ema is not None else model.state_dict()
            torch.save(
                {"config": asdict(cfg), "state_dict": state, "epoch": epoch, "val_mae": val_mae},
                out_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "lr": scheduler.get_last_lr()[0],
                "seconds": time.time() - epoch_start,
            }
        )
        marker = " *" if improved else ""
        print(
            f"epoch {epoch:3d} | train {train_mae * 1000:8.2f} | val {val_mae * 1000:8.2f} meV"
            f" | lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - epoch_start:6.1f}s{marker}",
            flush=True,
        )

        if cfg.patience and epochs_without_improvement >= cfg.patience:
            print(f"early stopping: no improvement for {cfg.patience} epochs")
            break

    # Test with the best checkpoint, never the last one.
    checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_mae = evaluate(model, test_loader, datamodule, device)

    result = {
        "run_name": run_name,
        "config": asdict(cfg),
        "num_params": num_params,
        "avg_num_neighbors": avg_neighbors,
        "splits": sizes,
        "best_epoch": best_epoch,
        "val_mae": best_val,
        "test_mae": test_mae,
        "val_mae_meV": best_val * 1000,
        "test_mae_meV": test_mae * 1000,
        "units": datamodule.units,
        "total_seconds": time.time() - start,
        "history": history,
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"\nbest epoch {best_epoch} | val {best_val * 1000:.2f} meV | "
          f"TEST {test_mae * 1000:.2f} meV | {(time.time() - start) / 60:.1f} min")
    return result


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    for name, value in asdict(defaults).items():
        if name == "extra":
            continue
        if isinstance(value, bool):
            # BooleanOptionalAction so that True-by-default flags can actually be
            # turned off (`--no-batch_norm`); `store_true` would make them permanent.
            parser.add_argument(
                f"--{name}", action=argparse.BooleanOptionalAction, default=value
            )
        elif value is None:
            parser.add_argument(f"--{name}", type=str, default=None)
        else:
            parser.add_argument(f"--{name}", type=type(value), default=value)
    args = parser.parse_args(argv)
    return TrainConfig(**{k: v for k, v in vars(args).items()})


if __name__ == "__main__":
    train(parse_args())
