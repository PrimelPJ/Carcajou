#!/usr/bin/env python3
"""Phase 1: the mask ablation, on the same harness as everything else.

    python scripts/run_vision_ablation.py                 # full, ~15 min
    python scripts/run_vision_ablation.py --seeds 1 --laps 2   # smoke, ~2 min

Adds VO rows to the consumer-MEMS outage table:

    ins+zupt+nhc                the existing baseline, re-run for byte parity
    ins+zupt+nhc+vo(mask off)   features on the lead vehicle reach RANSAC
    ins+zupt+nhc+vo(mask on)    realistic segmenter (recall 0.92, fpr 0.02)

Methodology is unchanged from run_benchmark.py: every ablation resumes from
the same snapshots of a single GNSS-aided pass (which itself never sees VO),
same IMU realisation, same windows. VO measurements are generated once per
(seed, mask) from truth geometry, so mask-on and mask-off differ only in which
correspondences the pose estimator saw.

Outputs
-------
results/vision_ablation.json    raw per-window results + VO diagnostics
results/vision_table.md         the ablation rows for the README
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from carcajou.benchmark.metrics import summarize  # noqa: E402
from carcajou.datasets.synthetic import (  # noqa: E402
    corrupt_imu,
    make_trajectory,
    perfect_imu,
    simulate_gnss,
)
from carcajou.eskf import EskfConfig  # noqa: E402
from carcajou.pipeline import index_vo, run_outage_study  # noqa: E402
from carcajou.sensors import GNSS_GRADES, IMU_GRADES  # noqa: E402
from carcajou.vision import (  # noqa: E402
    KITTI_LIKE,
    MASK_OFF,
    MASK_REALISTIC,
    StereoVo,
    VoConfig,
    make_world,
)

DEFAULT_DURATIONS = [10.0, 30.0, 60.0, 120.0]

ABLATIONS = {
    "ins+zupt+nhc": dict(use_zupt=True, use_nhc=True, use_vo=False),
    "ins+zupt+nhc+vo(mask off)": dict(use_zupt=True, use_nhc=True, use_vo=True),
    "ins+zupt+nhc+vo(mask on)": dict(use_zupt=True, use_nhc=True, use_vo=True),
}
MASKS = {
    "ins+zupt+nhc+vo(mask off)": MASK_OFF,
    "ins+zupt+nhc+vo(mask on)": MASK_REALISTIC,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--imu", default="consumer-mems", choices=sorted(IMU_GRADES))
    ap.add_argument("--gnss", default="spp", choices=sorted(GNSS_GRADES))
    ap.add_argument("--durations", nargs="+", type=float, default=DEFAULT_DURATIONS)
    ap.add_argument("--window-spacing", type=float, default=60.0)
    ap.add_argument("--warmup", type=float, default=120.0)
    ap.add_argument("--gap", type=float, default=12.0, help="lead vehicle gap, metres")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    traj = make_trajectory(laps=args.laps, rate_hz=args.rate)
    clean = perfect_imu(traj)
    durations = sorted(args.durations)
    warmup = min(args.warmup, float(traj.t[-1]) / 3.0)
    print(
        f"trajectory: {traj.t[-1]:.0f} s, {traj.arc_length()[-1]:.0f} m, "
        f"imu {args.imu}, {args.seeds} seeds"
    )

    imu_grade, gnss_grade = IMU_GRADES[args.imu], GNSS_GRADES[args.gnss]
    aided_cfg = EskfConfig(imu=imu_grade, gnss=gnss_grade, use_zupt=True, use_nhc=True)
    outage_cfgs = {
        k: EskfConfig(imu=imu_grade, gnss=gnss_grade, **v) for k, v in ABLATIONS.items()
    }

    pooled: dict[str, dict[float, list]] = {k: {d: [] for d in durations} for k in ABLATIONS}
    vo_diag: dict[str, list[dict]] = {k: [] for k in MASKS}
    raw: dict = {"meta": vars(args) | {"durations": durations, "warmup": warmup}, "runs": []}

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        rng = np.random.default_rng(1000 + seed)
        imus, _, _ = corrupt_imu(clean, imu_grade, rng, traj.dt)
        fixes = simulate_gnss(traj, gnss_grade, rng)

        # One landmark world per seed; masks see identical geometry.
        world = make_world(traj, np.random.default_rng(9000 + seed), gap_m=args.gap)

        t0 = time.time()
        vo_sources = {}
        for name, mask in MASKS.items():
            vo = StereoVo(
                KITTI_LIKE,
                world,
                traj,
                VoConfig(),
                mask=mask,
                rng=np.random.default_rng(5000 + seed),
            )
            meas = vo.run()
            vo_sources[name] = index_vo(meas)
            dyn_frac = [m.diagnostics["dynamic_inlier_fraction"] for m in meas]
            vo_diag[name].append(
                {
                    "seed": seed,
                    "n_epochs": len(meas),
                    "mean_inliers": float(np.mean([m.n_inliers for m in meas])),
                    "mean_dynamic_inlier_fraction": float(np.mean(dyn_frac)),
                    "p95_dynamic_inlier_fraction": float(np.percentile(dyn_frac, 95)),
                }
            )
        print(f"  seed {seed}: VO generated in {time.time() - t0:.0f} s")

        t0 = time.time()
        _, study = run_outage_study(
            traj,
            imus,
            fixes,
            aided_cfg,
            outage_cfgs,
            durations,
            window_spacing=args.window_spacing,
            warmup=warmup,
            vo_sources=vo_sources,
        )
        print(f"  seed {seed}: outage study in {time.time() - t0:.0f} s")

        for name, per_d in study.items():
            for d, results in per_d.items():
                pooled[name][d].extend(results)
                raw["runs"].append(
                    {
                        "seed": seed,
                        "ablation": name,
                        "duration": d,
                        "windows": [r.__dict__ for r in results],
                    }
                )

    # ------------------------------------------------------------- reporting
    lines = [
        f"**IMU: `{args.imu}`** with stereo VO aiding "
        f"({int(sum(len(v) for v in pooled[next(iter(ABLATIONS))].values()) / len(durations))} "
        "windows per cell)",
        "",
        "| aiding during outage | " + " | ".join(f"{int(d)} s" for d in durations) + " |",
        "|---" * (len(durations) + 1) + "|",
    ]
    summary: dict[str, dict] = {}
    for name in ABLATIONS:
        cells = []
        summary[name] = {}
        for d in durations:
            s = summarize(pooled[name][d])
            summary[name][str(int(d))] = s
            cells.append(f"{s['drift_pct_median']:.2f} / {s['drift_pct_p95']:.2f}")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print("\n" + table + "\n")

    for name, diags in vo_diag.items():
        dyn = np.mean([d["mean_dynamic_inlier_fraction"] for d in diags])
        inl = np.mean([d["mean_inliers"] for d in diags])
        print(
            f"{name}: mean inliers/epoch {inl:.0f}, "
            f"mean dynamic inlier fraction {dyn:.3f}"
        )

    raw["summary"] = summary
    raw["vo_diagnostics"] = vo_diag
    suffix = f"_s{args.seed_start}" if args.seeds == 1 else ""
    (outdir / f"vision_ablation{suffix}.json").write_text(json.dumps(raw, indent=2))
    (outdir / "vision_table.md").write_text(table + "\n")
    print(f"\nwrote {outdir}/vision_ablation.json and {outdir}/vision_table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
