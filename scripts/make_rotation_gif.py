"""Animate a molecule rotating, with each model's live prediction beside it.

The static figure proves the point numerically; this makes it immediate. As the molecule
spins, the equivariant model's needle does not move at all while the raw-coordinate
model's wanders visibly.

Written for the README, so it stays small: ~120 frames at modest dpi.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from symmetrynet.data.qm9 import QM9DataModule  # noqa: E402
from symmetrynet.models import NaiveCoordinateGNN, TensorFieldNetwork  # noqa: E402
from symmetrynet.scratch.wigner import rotation_matrix  # noqa: E402
from symmetrynet.utils.precision import default_dtype  # noqa: E402

DT = torch.float64

# Colours and radii roughly following CPK convention, indexed by atomic number.
ELEMENT_STYLE = {
    1: ("#e8e8e8", 90, "H"),
    6: ("#3d3d3d", 220, "C"),
    7: ("#2f5fd0", 240, "N"),
    8: ("#d63b2f", 240, "O"),
    9: ("#5fbf5f", 230, "F"),
}
COVALENT_RADIUS = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}


def find_bonds(pos: np.ndarray, z: np.ndarray, tolerance: float = 1.3) -> list[tuple[int, int]]:
    """Bond if the separation is under the scaled sum of covalent radii (drawing only)."""
    bonds = []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            limit = tolerance * (COVALENT_RADIUS.get(int(z[i]), 0.7)
                                 + COVALENT_RADIUS.get(int(z[j]), 0.7))
            if np.linalg.norm(pos[i] - pos[j]) < limit:
                bonds.append((i, j))
    return bonds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", type=int, default=1234)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--out", type=str, default=str(REPO_ROOT / "results" / "rotation.gif"))
    args = parser.parse_args()

    torch.manual_seed(0)
    dm = QM9DataModule(target="gap")
    dm.setup()
    data = dm.dataset[args.molecule]
    species = data.species
    pos = (data.pos.to(DT) - data.pos.to(DT).mean(0, keepdim=True))
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    z = data.z.numpy()

    avg_nb = dm.average_num_neighbors(5.0)
    with default_dtype(DT):
        equivariant = TensorFieldNetwork(
            multiplicity=32, l_max=2, num_layers=3, avg_num_neighbors=avg_nb
        ).eval()
        naive = NaiveCoordinateGNN(hidden=64, num_layers=3, avg_num_neighbors=avg_nb).eval()

    angles = np.linspace(0, 2 * np.pi, args.frames, endpoint=False)
    rotations, eq_pred, nv_pred, frames = [], [], [], []
    with torch.no_grad():
        for angle in angles:
            rot = rotation_matrix(
                torch.tensor(angle, dtype=DT),
                torch.tensor(0.55, dtype=DT),
                torch.tensor(0.0, dtype=DT),
            )
            rotations.append(rot)
            rotated = pos @ rot.T
            frames.append(rotated.numpy())
            eq_pred.append(equivariant(species, rotated, batch).item())
            nv_pred.append(naive(species, rotated, batch).item())

    eq_pred, nv_pred = np.array(eq_pred), np.array(nv_pred)
    bonds = find_bonds(pos.numpy(), z)
    limit = float(np.abs(np.stack(frames)).max()) * 1.15

    fig = plt.figure(figsize=(10.5, 5.0))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axp = fig.add_subplot(1, 2, 2)

    # Centre both traces on their own mean so the shapes, not the offsets, are compared.
    eq_c, nv_c = eq_pred - eq_pred.mean(), nv_pred - nv_pred.mean()
    span = max(np.abs(nv_c).max(), 1e-3) * 1.4

    axp.axhline(0.0, color="0.8", lw=0.8)
    (line_eq,) = axp.plot([], [], color="#1b6ca8", lw=2.6, label="Equivariant TFN")
    (line_nv,) = axp.plot([], [], color="#d1495b", lw=2.0, label="Naive GNN (raw coords)")
    (dot_eq,) = axp.plot([], [], "o", color="#1b6ca8", ms=7)
    (dot_nv,) = axp.plot([], [], "o", color="#d1495b", ms=7)
    axp.set_xlim(0, 2 * np.pi)
    axp.set_ylim(-span, span)
    axp.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    axp.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    axp.set_xlabel("rotation angle")
    axp.set_ylabel("prediction, mean-centred (eV)")
    axp.set_title("prediction while the molecule spins")
    axp.legend(fontsize=9, loc="upper right")
    axp.grid(alpha=0.25)

    readout = axp.text(0.03, 0.06, "", transform=axp.transAxes, fontsize=8.5,
                       family="monospace", va="bottom")

    def draw(frame: int):
        ax3d.clear()
        p = frames[frame]
        for i, j in bonds:
            ax3d.plot(*zip(p[i], p[j], strict=True), color="0.45", lw=2.0, zorder=1)
        for atomic_number in np.unique(z):
            color, size, _ = ELEMENT_STYLE.get(int(atomic_number), ("#999999", 150, "?"))
            mask = z == atomic_number
            ax3d.scatter(p[mask, 0], p[mask, 1], p[mask, 2], c=color, s=size,
                         edgecolors="0.25", linewidths=0.6, depthshade=True, zorder=2)
        ax3d.set_xlim(-limit, limit)
        ax3d.set_ylim(-limit, limit)
        ax3d.set_zlim(-limit, limit)
        ax3d.set_axis_off()
        ax3d.set_title("the same molecule, rotated", fontsize=11)

        line_eq.set_data(angles[: frame + 1], eq_c[: frame + 1])
        line_nv.set_data(angles[: frame + 1], nv_c[: frame + 1])
        dot_eq.set_data([angles[frame]], [eq_c[frame]])
        dot_nv.set_data([angles[frame]], [nv_c[frame]])
        readout.set_text(
            f"equivariant  drift {abs(eq_c[frame]):.2e} eV\n"
            f"naive        drift {abs(nv_c[frame]):.2e} eV"
        )
        return line_eq, line_nv, dot_eq, dot_nv, readout

    fig.suptitle("Physics says the prediction cannot change. Only one of these models agrees.",
                 fontsize=12)
    fig.tight_layout()

    animation = FuncAnimation(fig, draw, frames=args.frames, blit=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    animation.save(out, writer=PillowWriter(fps=args.fps), dpi=95)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  equivariant peak-to-peak: {np.ptp(eq_pred):.3e} eV")
    print(f"  naive       peak-to-peak: {np.ptp(nv_pred):.3e} eV")


if __name__ == "__main__":
    main()
