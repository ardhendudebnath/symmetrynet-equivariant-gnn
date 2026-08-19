"""Build the project site: gather every measured number, inline it, render the page.

Keeps the page honest by construction. Every figure it displays is read from the JSON
that the experiments actually wrote, so the site cannot drift away from the results the
way a hand-maintained page would.

Run ``scripts/make_interactive_demo.py`` first (it produces the orientation grid), then::

    python scripts/make_site.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
PLACEHOLDER = "__SITE_DATA__"


def load(name: str, default=None):
    path = RESULTS / name
    if not path.exists():
        print(f"  warning: {name} missing")
        return default
    return json.loads(path.read_text())


def build_payload() -> dict:
    demo = load("demo_data.json", {})
    grid = demo.get("grid", {})
    models = demo.get("models", {})

    # Subsample the orientation grid: the page only needs enough resolution for a smooth
    # readout while dragging, and the full 72x37 triples the page size for no benefit.
    step_a, step_e = 2, 2
    compact_models = {}
    for key, entry in models.items():
        values = entry["values"]
        compact_models[key] = {
            "values": [row[::step_a] for row in values[::step_e]],
            "spread": entry["spread"],
        }
    compact_grid = {
        "n_azimuth": len(next(iter(compact_models.values()))["values"][0]) if compact_models else 0,
        "n_elevation": len(next(iter(compact_models.values()))["values"]) if compact_models else 0,
        "azimuth_step": grid.get("azimuth_step", 0) * step_a,
        "elevation_start": grid.get("elevation_start", 0),
        "elevation_step": grid.get("elevation_step", 0) * step_e,
    }

    comparison = {r["model"]: r for r in (load("comparison.json") or [])}
    ablation = sorted(load("ablation.json") or [], key=lambda r: r["l_max"] or 0)
    scaling = load("scaling.json", {})
    multiseed = load("multiseed_analysis.json", {})
    md17 = load("md17_forces.json", {})
    irrep = load("irrep_utilization.json", {})
    equivariance = load("equivariance.json", {})

    return {
        "molecule": demo.get("molecule", {}),
        "grid": compact_grid,
        "models": compact_models,
        "comparison": [
            {
                "key": key,
                "test_mae": comparison[key]["test_mae_meV"],
                "params": comparison[key]["num_params"],
            }
            for key in ("painn", "angular", "baseline", "tfn")
            if key in comparison
        ],
        "ablation": [
            {"l_max": r["l_max"], "test_mae": r["test_mae_meV"], "params": r["num_params"]}
            for r in ablation
        ],
        "scaling": scaling.get("per_model", {}),
        "scaling_delta": scaling.get("exponent_difference", {}),
        "multiseed": multiseed.get("per_fraction", []),
        "md17": md17.get("points", []),
        "irrep": irrep.get("per_degree", []),
        "equivariance": equivariance.get("worst_relative_deviation", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=str(REPO_ROOT / "scripts" / "site_template.html"))
    parser.add_argument("--out", default=str(RESULTS / "site.html"))
    args = parser.parse_args()

    payload = build_payload()
    template = Path(args.template).read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise SystemExit(f"template is missing {PLACEHOLDER}")

    # Escape `</` so the JSON can never close the enclosing <script> early.
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    out = Path(args.out)
    out.write_text(template.replace(PLACEHOLDER, blob), encoding="utf-8")

    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"  models in comparison : {[r['key'] for r in payload['comparison']]}")
    print(f"  ablation points      : {len(payload['ablation'])}")
    print(f"  multiseed fractions  : {len(payload['multiseed'])}")
    print(f"  md17 points          : {len(payload['md17'])}")


if __name__ == "__main__":
    main()
