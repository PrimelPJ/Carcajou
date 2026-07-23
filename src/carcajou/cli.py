"""Console entry point: ``carcajou-bench``."""

from __future__ import annotations

import pathlib
import runpy
import sys


def main() -> int:
    script = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "run_benchmark.py"
    if not script.exists():  # installed without the repo tree
        print("run_benchmark.py not found; run from a source checkout", file=sys.stderr)
        return 1
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")
    return 0
