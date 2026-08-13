"""Precompute the data for the interactive rotation demo.

The demo lets a reader drag a molecule around and watch two predictions: the equivariant
model's, which does not move, and a raw-coordinate model's, which does. Running a GNN in
the browser is not worth the trouble, so instead we evaluate both models over a grid of
orientations here and embed the results; the page just rotates the coordinates and looks
the numbers up.

Orientations are parameterised by ``R = Rz(azimuth) @ Ry(elevation)``, which is trivial to
reproduce in JavaScript so the rendered geometry and the looked-up prediction always agree.

Writes ``results/demo_data.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from symmetrynet.data.qm9 import QM9DataModule  # noqa: E402
from symmetrynet.models import (  # noqa: E402
    InvariantGNN,
    NaiveCoordinateGNN,
    TensorFieldNetwork,
)
from symmetrynet.utils.precision import default_dtype  # noqa: E402

DT = torch.float64

ELEMENTS = {
    1: {"symbol": "H", "color": "#e6e6e6", "radius": 0.32},
    6: {"symbol": "C", "color": "#4a4a4a", "radius": 0.72},
    7: {"symbol": "N", "color": "#3b6fd4", "radius": 0.68},
    8: {"symbol": "O", "color": "#d64545", "radius": 0.66},
    9: {"symbol": "F", "color": "#4aa564", "radius": 0.58},
}
COVALENT = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}


def rotation(azimuth: float, elevation: float) -> torch.Tensor:
    """``Rz(azimuth) @ Ry(elevation)`` -- kept simple so JS can mirror it exactly."""
    ca, sa = np.cos(azimuth), np.sin(azimuth)
    ce, se = np.cos(elevation), np.sin(elevation)
    rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[ce, 0.0, se], [0.0, 1.0, 0.0], [-se, 0.0, ce]])
    return torch.tensor(rz @ ry, dtype=DT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", type=int, default=1234)
    parser.add_argument("--n-azimuth", type=int, default=72)
    parser.add_argument("--n-elevation", type=int, default=37)
    parser.add_argument("--out", type=str, default=str(REPO_ROOT / "results" / "demo_data.json"))
    args = parser.parse_args()

    torch.manual_seed(0)
    dm = QM9DataModule(target="gap")
    dm.setup()
    data = dm.dataset[args.molecule]
    pos = data.pos.to(DT)
    pos = pos - pos.mean(0, keepdim=True)
    species, z = data.species, data.z
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    avg_nb = dm.average_num_neighbors(5.0)

    with default_dtype(DT):
        models = {
            "equivariant": TensorFieldNetwork(
                multiplicity=32, l_max=2, num_layers=3, avg_num_neighbors=avg_nb
            ).eval(),
            "invariant": InvariantGNN(
                hidden=64, num_layers=3, avg_num_neighbors=avg_nb
            ).eval(),
            "naive": NaiveCoordinateGNN(
                hidden=64, num_layers=3, avg_num_neighbors=avg_nb
            ).eval(),
        }

    azimuths = np.linspace(0.0, 2 * np.pi, args.n_azimuth, endpoint=False)
    elevations = np.linspace(-np.pi / 2, np.pi / 2, args.n_elevation)

    grids = {name: np.zeros((args.n_elevation, args.n_azimuth)) for name in models}
    with torch.no_grad():
        for i, elevation in enumerate(elevations):
            for j, azimuth in enumerate(azimuths):
                rotated = pos @ rotation(float(azimuth), float(elevation)).T
                for name, model in models.items():
                    grids[name][i, j] = model(species, rotated, batch).item()
        print(f"evaluated {args.n_elevation * args.n_azimuth} orientations x {len(models)} models")

    # Bonds are for drawing only; distance-based detection is plenty.
    zs = z.tolist()
    bonds = []
    for a in range(len(zs)):
        for b in range(a + 1, len(zs)):
            limit = 1.3 * (COVALENT.get(zs[a], 0.7) + COVALENT.get(zs[b], 0.7))
            if float((pos[a] - pos[b]).norm()) < limit:
                bonds.append([a, b])

    formula = "".join(
        f"{ELEMENTS[el]['symbol']}{zs.count(el)}" for el in (6, 1, 7, 8, 9) if zs.count(el)
    )

    payload = {
        "molecule": {
            "index": args.molecule,
            "formula": formula,
            "atoms": [
                {
                    "z": zval,
                    "symbol": ELEMENTS[zval]["symbol"],
                    "color": ELEMENTS[zval]["color"],
                    "radius": ELEMENTS[zval]["radius"],
                    "pos": [round(float(c), 5) for c in pos[k]],
                }
                for k, zval in enumerate(zs)
            ],
            "bonds": bonds,
        },
        "grid": {
            "n_azimuth": args.n_azimuth,
            "n_elevation": args.n_elevation,
            "azimuth_start": 0.0,
            "azimuth_step": float(2 * np.pi / args.n_azimuth),
            "elevation_start": float(-np.pi / 2),
            "elevation_step": float(np.pi / (args.n_elevation - 1)),
        },
        "models": {},
    }

    for name, grid in grids.items():
        # Store the *deviation* from the mean, not the absolute prediction.
        #
        # This is not a size optimisation, it is a precision requirement.  An untrained
        # network can output values of order 100 eV while its rotational variation is
        # order 1e-13 eV.  Rounding absolute values to any reasonable number of decimal
        # places quantises that variation away entirely, and the page would then display
        # a flat, dishonest 0.0 for the equivariant model instead of the true 1e-13.
        # Deviations are small numbers, so significant figures are preserved.
        # The demo only ever displays drift, in which the mean cancels anyway.
        deviation = grid - grid.mean()
        payload["models"][name] = {
            "values": [[float(f"{v:.6g}") for v in row] for row in deviation],
            "mean": float(grid.mean()),
            "spread": float(grid.max() - grid.min()),
        }
        print(f"  {name:<12s} mean {grid.mean():+.5f} eV   peak-to-peak {np.ptp(grid):.3e} eV")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
