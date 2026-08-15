r"""How much does each irrep actually contribute to a trained model's prediction?

The question this answers
-------------------------
The ablation in this project shows ``l_max=2`` beating ``l_max=1`` by 12% inside the TFN
family, which says higher angular resolution helps. Yet PaiNN, structurally incapable of
representing anything above :math:`\ell = 1`, beats TFN-at-``l_max=2`` by 27%. Both
statements are measured, and together they are confusing: is the TFN's :math:`\ell=2`
channel doing real work, or is it mostly along for the ride while the extra parameters and
5x compute do the lifting?

Model accuracy alone cannot separate those. This measures the channels directly.

Two probes, after two failed ones
---------------------------------
The obvious measurements do not work, and it is worth saying why before the ones that do.

*Zero a degree's channels and measure the damage.* Uninformative: removing any degree
degrades test MAE by ~1800%, including :math:`\ell=0`. Deleting an entire channel takes a
deep network so far off its training distribution that everything collapses equally, so
the probe cannot discriminate between degrees.

*Fraction of total feature magnitude per degree.* Measured 13.8 / 31.8 / 54.4% for
:math:`\ell = 0,1,2`. But those are almost exactly the *dimension* shares -- 11.1 / 33.3 /
55.6%, since a degree-:math:`\ell` channel holds :math:`2\ell+1` components. The number was
reporting how many slots each degree owns, not how hard it is working.

What is used instead:

**Magnitude per component.** The same norms divided by the number of components, which
removes the dimensionality confound. A degree whose per-component magnitude matches the
others is neither amplified nor suppressed relative to its size.

**Gradient sensitivity.** The mean magnitude of :math:`\partial \hat y / \partial h_\ell`,
i.e. how strongly the prediction responds to a perturbation of each degree's features. This
is causal like ablation but does not leave the data distribution: it is a local derivative
at the operating point rather than a demolition. Also reported per component.

Neither is decisive alone; agreement between them is the signal worth trusting.

Usage::

    python scripts/irrep_utilization.py --run tfn_l2_f1_e200_s0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from symmetrynet.data.qm9 import QM9DataModule  # noqa: E402
from symmetrynet.train import TrainConfig, build_model, evaluate  # noqa: E402

DEFAULT_RUNS = Path.home() / ".symmetrynet" / "runs"


def irrep_slices(irreps) -> dict[int, list[tuple[int, int]]]:
    """Degree -> the column ranges it occupies in a flattened e3nn feature tensor."""
    spans: dict[int, list[tuple[int, int]]] = {}
    offset = 0
    for mul, ir in irreps:
        width = mul * ir.dim
        spans.setdefault(ir.l, []).append((offset, offset + width))
        offset += width
    return spans


class DegreeAblator:
    """Zeroes every channel of one degree on the output of each interaction block."""

    def __init__(self, model, degree: int):
        self.handles = []
        self.degree = degree
        for layer in model.layers:
            spans = irrep_slices(layer.irreps_out).get(degree)
            if not spans:
                continue
            self.handles.append(layer.register_forward_hook(self._make_hook(spans)))

    @staticmethod
    def _make_hook(spans):
        def hook(_module, _inputs, output):
            out = output.clone()
            for start, stop in spans:
                out[..., start:stop] = 0.0
            return out

        return hook

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


def measure(model, loader, device: str, max_batches: int = 20) -> dict[int, dict]:
    """Per-component feature magnitude and prediction sensitivity, per degree.

    Both quantities are divided by the number of components a degree occupies, since a
    degree-l channel holds 2l+1 of them; without that division the numbers just recover
    the dimension split.
    """
    sq_sum: dict[int, float] = {}
    grad_sum: dict[int, float] = {}
    counts: dict[int, int] = {}
    dims: dict[int, int] = {}

    stash: list[tuple[torch.Tensor, dict[int, list[tuple[int, int]]]]] = []

    def make_hook(layer):
        spans = irrep_slices(layer.irreps_out)

        def hook(_module, _inputs, output):
            output.retain_grad()
            stash.append((output, spans))

        return hook

    handles = [layer.register_forward_hook(make_hook(layer)) for layer in model.layers]
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        stash.clear()
        batch = batch.to(device)
        prediction = model(batch.species, batch.pos, batch.batch)
        # Gradient of the prediction w.r.t. every intermediate feature tensor at once.
        model.zero_grad(set_to_none=True)
        prediction.sum().backward()

        for tensor, spans in stash:
            for degree, ranges in spans.items():
                for start, stop in ranges:
                    # detach: these are read-outs, and float() on a grad-tracking tensor
                    # both warns and needlessly keeps the graph alive.
                    block = tensor[..., start:stop].detach()
                    width = stop - start
                    sq_sum[degree] = sq_sum.get(degree, 0.0) + float(block.pow(2).sum())
                    if tensor.grad is not None:
                        grad_block = tensor.grad[..., start:stop]
                        grad_sum[degree] = grad_sum.get(degree, 0.0) + float(
                            grad_block.abs().sum()
                        )
                    counts[degree] = counts.get(degree, 0) + block.shape[0] * width
                    dims[degree] = 2 * degree + 1

    for handle in handles:
        handle.remove()

    out = {}
    for degree in sorted(sq_sum):
        n = max(counts.get(degree, 1), 1)
        out[degree] = {
            "rms_per_component": (sq_sum[degree] / n) ** 0.5,
            "grad_per_component": grad_sum.get(degree, 0.0) / n,
            "components": dims.get(degree, 2 * degree + 1),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tfn_l2_f1_e200_s0")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "irrep_utilization.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.runs_dir) / args.run / "best.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"no checkpoint at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**checkpoint["config"])
    if cfg.model != "tfn":
        raise SystemExit(f"this analysis targets the e3nn TFN; {args.run} is {cfg.model!r}")

    datamodule = QM9DataModule(
        target=cfg.target, root=cfg.data_root, batch_size=cfg.batch_size, seed=cfg.seed
    )
    datamodule.setup()
    model = build_model(cfg, datamodule.average_num_neighbors(cfg.cutoff)).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    test_loader = datamodule.test_loader()
    baseline_mae = evaluate(model, test_loader, datamodule, args.device) * 1000
    print(f"run        : {args.run}  (l_max={cfg.l_max})")
    print(f"test MAE   : {baseline_mae:.2f} meV\n")

    stats = measure(model, test_loader, args.device)
    ref_rms = stats[0]["rms_per_component"] or 1.0
    ref_grad = stats[0]["grad_per_component"] or 1.0

    print(f"{'degree':>7s} {'comps':>6s} {'RMS/comp':>10s} {'vs l=0':>8s} "
          f"{'|grad|/comp':>12s} {'vs l=0':>8s}")
    print("-" * 56)
    rows = []
    for degree, s in stats.items():
        rows.append({"degree": degree, **s,
                     "rms_relative_to_l0": s["rms_per_component"] / ref_rms,
                     "grad_relative_to_l0": s["grad_per_component"] / ref_grad})
        print(f"{degree:>7d} {s['components']:>6d} {s['rms_per_component']:>10.4f} "
              f"{s['rms_per_component'] / ref_rms:>7.2f}x {s['grad_per_component']:>12.2e} "
              f"{s['grad_per_component'] / ref_grad:>7.2f}x")

    print("\nBoth columns are per component, so a degree is not credited merely for")
    print("occupying 2l+1 slots. Gradient sensitivity is how strongly the prediction")
    print("responds to perturbing that degree, measured at the operating point.")

    high = [r for r in rows if r["degree"] >= 2]
    if high:
        weakest = min(r["grad_relative_to_l0"] for r in high)
        if weakest < 0.5:
            print(f"\n  The prediction is {1 / weakest:.1f}x less sensitive to l>=2 features")
            print("  than to scalars, per component. The paths that require Clebsch-Gordan")
            print("  tensor products -- and cost 5x the compute -- carry proportionally")
            print("  less influence, which is consistent with PaiNN (l<=1) winning.")
        elif weakest > 0.9:
            print("\n  l>=2 features are as influential per component as scalars, so the")
            print("  TFN's loss to PaiNN is not explained by unused angular capacity.")

    Path(args.out).write_text(json.dumps(
        {"run": args.run, "l_max": cfg.l_max, "test_mae_meV": baseline_mae,
         "note": "per-component values; ablation was tried and discarded as uninformative",
         "per_degree": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
