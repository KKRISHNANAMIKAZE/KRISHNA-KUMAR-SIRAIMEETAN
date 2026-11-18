"""Shared helpers for the benchmark scripts (timing and environment info)."""

import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmark" / "results"
DATA = ROOT / "data" / "elec.csv"

# Let the scripts import the package when run as `python benchmark/<file>.py`.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def median_time(func, repeats=5, warmup=1):
    """Median wall-clock seconds of ``func()`` over ``repeats`` runs.

    One untimed warm-up run first, so one-off costs (imports, memory
    allocation, CPU caches) do not distort the first measurement.
    """
    for _ in range(warmup):
        func()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def environment():
    """Text describing the machine, so results can be reproduced."""
    return (f"Python {platform.python_version()} | NumPy {np.__version__} | "
            f"{platform.system()} {platform.release()} | "
            f"{platform.machine()} | {platform.processor() or 'unknown CPU'}")


def write_csv(path, header, rows):
    """Write rows of numbers/strings to a small CSV file."""
    lines = [",".join(header)]
    lines += [",".join(f"{v:.6g}" if isinstance(v, float) else str(v)
                       for v in row) for row in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
