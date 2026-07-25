"""Console entry point: ``carcajou-bench``."""

from __future__ import annotations

import pathlib
import runpy
import sys


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] in ("--version", "-V"):
        from carcajou import __version__
        print(f"carcajou {__version__}")
        return 0

    script = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "run_benchmark.py"
    if not script.exists():  # installed without the repo tree
        print("run_benchmark.py not found; run from a source checkout", file=sys.stderr)
        return 1
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")
    return 0
