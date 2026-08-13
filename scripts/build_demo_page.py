"""Inline the precomputed orientation data into the demo template.

Kept as a separate step so ``scripts/demo_template.html`` stays readable and editable in
the repository, rather than being a single unmaintainable file with 100 KB of numbers
pasted into the middle of it.

Run ``scripts/make_interactive_demo.py`` first to produce the data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = "__DEMO_DATA__"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=str(REPO_ROOT / "scripts" / "demo_template.html"))
    parser.add_argument("--data", default=str(REPO_ROOT / "results" / "demo_data.json"))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "demo.html"))
    args = parser.parse_args()

    template = Path(args.template).read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise SystemExit(f"template is missing the {PLACEHOLDER} placeholder")

    payload = json.loads(Path(args.data).read_text(encoding="utf-8"))
    # Compact separators keep the page small; `</` is escaped so the JSON can never
    # terminate the enclosing <script> tag early.
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template.replace(PLACEHOLDER, blob), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
