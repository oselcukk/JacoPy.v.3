"""Smoke-run every example notebook's code cells in order.

Each notebook gets a fresh namespace; the first failing cell aborts
that notebook and marks the run failed. This is the executable check
the Makefile's ``notebooks`` target pointed at a non-existent
``tests/test_docs`` for (2026-09-07 audit, maintenance note); it runs
code only — no Jupyter UI, no output-layout checks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text())
    namespace: dict = {}
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            exec(compile(source, f"{path.name}:cell{idx}", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - report and fail
            print(f"FAIL {path.name} cell {idx}: {type(exc).__name__}: {exc}")
            return False
    print(f"ok   {path.name}")
    return True


def main() -> int:
    notebooks = sorted((ROOT / "examples").rglob("*.ipynb"))
    if not notebooks:
        print("no notebooks found under examples/")
        return 1
    failures = sum(not run_notebook(nb) for nb in notebooks)
    print(f"{len(notebooks) - failures}/{len(notebooks)} notebooks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
