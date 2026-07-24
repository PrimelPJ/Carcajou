#!/usr/bin/env python3
"""Phase 2: scan-to-map localization on the outage harness.

    python scripts/run_lidar_ablation.py --seeds 1 --seed-start 0   # one shard

Rows added to the consumer-MEMS outage table:

    ins+zupt+nhc                    baseline, re-run for byte parity
    ins+zupt+nhc+map                map built with the segmentation mask
    ins+zupt+nhc+map(phantom)       identical pipeline, mask off at map time

The third row is the Phase 2 ablation that matters. Both map rows localize
identically, with identical trimming and identical acceptance gates; the only
difference is whether dynamic returns were removed when the map was *built*.
A lead vehicle baked into the map is a phantom that every later drive tries
to localize against. That is the empirical content of "AI-labelled HD map":
the labelling earns its keep at map-build time, and this measures how much.

Methodology is the Phase 0/1 harness unchanged: all ablations resume from the
same snapshots of one GNSS-aided pass. The map is built from that same pass's
*estimated* poses (never truth), then each outage window localizes against it.
The mapping and localization drives share sensor hardware but not noise:
scan noise is redrawn for localization, as a second drive would experience.
"""

from __future__ import annotations

import argparse
import dataclasses
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
from carcajou.lidar import LidarScanner, LidarSpec, build_map  # noqa: E402
from carcajou.lidar.matcher import MapMatcher  # noqa: E402
from carcajou.pipeline import run_aided_pass, run_outage_study  # noqa: E402
from carcajou.sensors import GNSS_GRADES, IMU_GRADES  # noqa: E402
from carcajou.vision import MASK_OFF, MASK_REALISTIC, make_world  # noqa: E402

DEFAULT_DURATIONS = [10.0, 30.0, 60.0, 120.0]

BASE = "ins+zupt+nhc"
MAP_MASKED = "ins+zupt+nhc+map"
MAP_PHANTOM = "ins+zupt+nhc+map(phantom)"


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
    ap.add_argument("--per-station", type=int, default=8)
    ap.add_argument("--gap-loc", type=float, default=24.0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    traj = make_trajectory(laps=args.laps, rate_hz=args.rate)
    clean = perfect_imu(traj)
    durations = sorted(args.durations)
    warmup = min(args.warmup, float(traj.t[-1]) / 3.0)

    imu_g, gnss_g = IMU_GRADES[args.imu], GNSS_GRADES[args.gnss]
    aided_cfg = EskfConfig(imu=imu_g, gnss=gnss_g, use_zupt=True, use_nhc=True)
    mk = lambda **kw: EskfConfig(  # noqa: E731
        imu=imu_g, gnss=gnss_g, use_zupt=True, use_nhc=True, **kw
    )
    cfgs = {BASE: mk(), MAP_MASKED: mk(use_map=True), MAP_PHANTOM: mk(use_map=True)}

    pooled = {n: {d: [] for d in durations} for n in cfgs}
    raw = {"meta": vars(args) | {"durations": durations, "warmup": warmup}, "runs": []}

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        rng = np.random.default_rng(1000 + seed)
        imus, _, _ = corrupt_imu(clean, imu_g, rng, traj.dt)
        fixes = simulate_gnss(traj, gnss_g, rng)
        world = make_world(
            traj, np.random.default_rng(9000 + seed), per_station=args.per_station
        )

        # ---- Pass 1: mapping. Scans posed at the aided filter's estimate.
        t0 = time.time()
        sc_map = LidarScanner(LidarSpec(), world, traj, rng=np.random.default_rng(7000 + seed))
        pose_idx = set(int(k) for k in sc_map.step_indices)
        # Snapshot placement uses the full-benchmark horizon, not this
        # shard's durations, so every shard (and every ablation) scores the
        # identical window set and rows remain comparable cell-for-cell.
        horizon = max(DEFAULT_DURATIONS)
        starts = np.arange(warmup, float(traj.t[-1]) - horizon - 1.0, args.window_spacing)
        pr = run_aided_pass(
            traj, imus, fixes, aided_cfg,
            snapshot_times=starts, capture_pose_indices=pose_idx,
        )
        p_bl = sc_map.spec.p_bl
        sensor_poses = {k: (R, p + R @ p_bl) for k, (R, p) in pr.poses.items()}
        scans = [sc_map.scan(k, noisy=True) for k in sorted(sensor_poses)]
        maps = {
            MAP_MASKED: build_map(
                scans, sensor_poses, MASK_REALISTIC, np.random.default_rng(200 + seed)
            ),
            MAP_PHANTOM: build_map(
                scans, sensor_poses, MASK_OFF, np.random.default_rng(200 + seed)
            ),
        }
        print(
            f"seed {seed}: maps built in {time.time() - t0:.0f} s "
            f"({len(maps[MAP_MASKED].points)} / {len(maps[MAP_PHANTOM].points)} voxels)"
        )

        # ---- Pass 2: localization during outages, fresh scan noise AND
        # different traffic: same static world, but the lead vehicle now
        # holds a different gap. This is the condition under which a phantom
        # matters. With identical traffic on both passes, a car baked into
        # the map is consistent with the live scan and the phantom row
        # measures nothing; with traffic moved, the live vehicle's returns
        # can alias onto the phantom ribbon at the wrong offset, and the map
        # voxels spent on the phantom mismatch live static structure. Maps
        # are for the world that stays; this is what happens when they also
        # contain the world that left.
        world_loc = dataclasses.replace(world, gap_m=args.gap_loc)
        matchers = {
            name: MapMatcher(
                LidarScanner(LidarSpec(), world_loc, traj, rng=np.random.default_rng(8000 + seed)),
                lmap,
                mask=MASK_OFF,
                rng=np.random.default_rng(300 + seed),
            )
            for name, lmap in maps.items()
        }
        t0 = time.time()
        _, study = run_outage_study(
            traj, imus, fixes, aided_cfg, cfgs, durations,
            window_spacing=args.window_spacing, warmup=warmup,
            matcher_sources=matchers, pass_result=pr,
        )
        print(f"seed {seed}: outage study in {time.time() - t0:.0f} s")

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

    n_win = len(pooled[BASE][durations[0]])
    lines = [
        f"**IMU: `{args.imu}`** with scan-to-map localization ({n_win} windows per cell)",
        "",
        "| aiding during outage | " + " | ".join(f"{int(d)} s" for d in durations) + " |",
        "|---" * (len(durations) + 1) + "|",
    ]
    summary = {}
    for name in cfgs:
        cells, summary[name] = [], {}
        for d in durations:
            s = summarize(pooled[name][d])
            summary[name][str(int(d))] = s
            cells.append(f"{s['drift_pct_median']:.2f} / {s['drift_pct_p95']:.2f}")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print("\n" + table)

    raw["summary"] = summary
    suffix = (f"_s{args.seed_start}" if args.seeds == 1 else "") + (
        f"_d{int(durations[0])}" if len(durations) < 4 else ""
    )
    (outdir / f"lidar_ablation{suffix}.json").write_text(json.dumps(raw, indent=1))
    (outdir / "lidar_table.md").write_text(table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
