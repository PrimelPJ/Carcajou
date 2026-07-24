#!/usr/bin/env python3
"""Phase 3: covariance consistency, 15-state vs 24-state.

    python scripts/run_consistency.py [--seeds 4] [--laps 2] [--imu industrial-mems]

Realistic errors ON (scale factor injected, GNSS position error split into
white + first-order Gauss-Markov), then the same aided run scored two ways:
the filter that pretends those errors do not exist (15 states) and the filter
that carries them (24 = 15 + accel scale, gyro scale, GNSS GM position).

Scored claim: fraction of epochs, after convergence, where true horizontal
error lies inside the filter's own reported 3-sigma bound. Nominal is 99.7 %.
A filter far below that is lying about its confidence, which is the defect
the README has documented since Phase 0.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from carcajou.datasets.synthetic import corrupt_imu, make_trajectory, perfect_imu, simulate_gnss
from carcajou.eskf import EskfConfig
from carcajou.pipeline import run_aided_pass
from carcajou.sensors import GNSS_GRADES, IMU_GRADES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--imu", default="industrial-mems", choices=sorted(IMU_GRADES))
    ap.add_argument("--gnss", default="spp-gm", choices=sorted(GNSS_GRADES))
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    traj = make_trajectory(laps=args.laps, rate_hz=100.0)
    clean = perfect_imu(traj)
    imu_g, gnss_g = IMU_GRADES[args.imu], GNSS_GRADES[args.gnss]

    out = {}
    for name, ext in (("15-state", False), ("24-state", True)):
        ins, rmss = [], []
        for seed in range(args.seeds):
            rng = np.random.default_rng(100 + seed)
            imus, _, _ = corrupt_imu(clean, imu_g, rng, traj.dt, inject_scale=True)
            fixes = simulate_gnss(traj, gnss_g, rng)
            cfg = EskfConfig(
                imu=imu_g, gnss=gnss_g, use_zupt=True, use_nhc=True, extended_state=ext
            )
            pr = run_aided_pass(traj, imus, fixes, cfg)
            err = np.linalg.norm((pr.p - traj.p)[:, :2], axis=1)
            bound = 3.0 * np.linalg.norm(pr.sigma_p[:, :2], axis=1)
            sl = slice(3000, None)
            ins.append(float(np.mean(err[sl] <= bound[sl])))
            rmss.append(float(np.sqrt(np.mean(err[sl] ** 2))))
        out[name] = {
            "pct_inside_3sigma": 100.0 * float(np.mean(ins)),
            "horiz_rms_m": float(np.mean(rmss)),
        }
        print(
            f"{name}: {out[name]['pct_inside_3sigma']:.1f} % inside 3-sigma "
            f"(nominal 99.7), horiz RMS {out[name]['horiz_rms_m']:.2f} m"
        )

    pathlib.Path(args.out).mkdir(exist_ok=True)
    (pathlib.Path(args.out) / "consistency.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
