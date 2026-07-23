#!/usr/bin/env python3
"""Run the GNSS-outage drift benchmark and emit results/ artefacts.

    python scripts/run_benchmark.py --laps 5 --seeds 5

Outputs
-------
results/benchmark.json    raw per-window results
results/table.md          the ablation table for the README
results/*.png             trajectory, error-growth and covariance plots
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import asdict

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
from carcajou.pipeline import run_outage_study  # noqa: E402
from carcajou.sensors import GNSS_GRADES, IMU_GRADES  # noqa: E402

DEFAULT_DURATIONS = [10.0, 30.0, 60.0, 120.0]

# What survives the outage. GNSS is off in every one of these by construction;
# the question is how much a system can recover from sources that cost nothing.
ABLATIONS = {
    "ins-only": dict(use_zupt=False, use_nhc=False),
    "ins+zupt": dict(use_zupt=True, use_nhc=False),
    "ins+nhc": dict(use_zupt=False, use_nhc=True),
    "ins+zupt+nhc": dict(use_zupt=True, use_nhc=True),
}


def build_cfgs(imu_name: str, gnss_name: str):
    imu, gnss = IMU_GRADES[imu_name], GNSS_GRADES[gnss_name]
    aided = EskfConfig(imu=imu, gnss=gnss, use_zupt=True, use_nhc=True)
    outage = {k: EskfConfig(imu=imu, gnss=gnss, **v) for k, v in ABLATIONS.items()}
    return aided, outage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--gnss", default="spp", choices=sorted(GNSS_GRADES))
    ap.add_argument(
        "--imu", nargs="+", default=["consumer-mems", "industrial-mems", "tactical"]
    )
    ap.add_argument(
        "--durations",
        nargs="+",
        type=float,
        default=DEFAULT_DURATIONS,
        help="GNSS outage lengths to score, seconds",
    )
    ap.add_argument("--window-spacing", type=float, default=60.0)
    ap.add_argument(
        "--warmup",
        type=float,
        default=120.0,
        help="settling time before the first outage window; clamped to a "
        "third of the trajectory so short runs still produce windows",
    )
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    traj = make_trajectory(laps=args.laps, rate_hz=args.rate)
    clean = perfect_imu(traj)
    durations = sorted(args.durations)
    warmup = min(args.warmup, float(traj.t[-1]) / 3.0)
    print(
        f"trajectory: {traj.t[-1]:.0f} s, {traj.arc_length()[-1]:.0f} m, "
        f"{len(clean)} IMU epochs at {args.rate:.0f} Hz"
    )

    raw: dict = {"meta": vars(args) | {"durations": durations, "warmup": warmup}, "runs": []}
    agg: dict[str, dict[str, dict[str, dict]]] = {}
    keep_pass = None

    for imu_name in args.imu:
        agg[imu_name] = {}
        pooled: dict[str, dict[float, list]] = {k: {d: [] for d in durations} for k in ABLATIONS}

        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            imus, _, _ = corrupt_imu(clean, IMU_GRADES[imu_name], rng, traj.dt)
            fixes = simulate_gnss(traj, GNSS_GRADES[args.gnss], rng)
            aided_cfg, outage_cfgs = build_cfgs(imu_name, args.gnss)

            t0 = time.time()
            pass_result, study = run_outage_study(
                traj,
                imus,
                fixes,
                aided_cfg,
                outage_cfgs,
                durations,
                window_spacing=args.window_spacing,
                warmup=warmup,
            )
            print(
                f"  {imu_name:<16} seed {seed}  "
                f"{len(pass_result.snapshots)} windows  {time.time() - t0:5.1f}s"
            )

            if keep_pass is None:
                keep_pass = (pass_result, traj, imus, fixes, aided_cfg, outage_cfgs)

            for name, by_dur in study.items():
                for d, res in by_dur.items():
                    pooled[name][d].extend(res)
                    raw["runs"].extend(
                        {"imu": imu_name, "ablation": name, "seed": seed, **asdict(r)}
                        for r in res
                    )

        for name, by_dur in pooled.items():
            agg[imu_name][name] = {str(int(d)): summarize(r) for d, r in by_dur.items()}

    raw["summary"] = agg
    (outdir / "benchmark.json").write_text(json.dumps(raw, indent=2))

    table = render_table(agg, args, durations)
    (outdir / "table.md").write_text(table)
    print("\n" + table)

    if not args.no_plots and keep_pass is not None:
        make_plots(outdir, *keep_pass)
    return 0


def render_table(agg: dict, args, durations: list[float]) -> str:
    lines = [
        "### GNSS-outage drift, percent of distance travelled (median / p95)",
        "",
        f"GNSS grade `{args.gnss}` before the outage. "
        f"{args.seeds} sensor-noise seeds per configuration. "
        f"Target: **< 1.0 %**.",
        "",
    ]
    for imu_name, by_abl in agg.items():
        lines += [
            f"**IMU: `{imu_name}`**",
            "",
            "| aiding during outage | " + " | ".join(f"{int(d)} s" for d in durations) + " |",
            "|---" * (len(durations) + 1) + "|",
        ]
        for name, by_dur in by_abl.items():
            cells = []
            for d in durations:
                s = by_dur[str(int(d))]
                cells.append(f"{s['drift_pct_median']:.2f} / {s['drift_pct_p95']:.2f}")
            lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def make_plots(outdir, pass_result, traj, imus, fixes, aided_cfg, outage_cfgs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

    # 1. trajectory
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(traj.p[:, 1], traj.p[:, 0], lw=1.6, label="ground truth")
    ax.plot(pass_result.p[:, 1], pass_result.p[:, 0], lw=0.9, ls="--", label="GNSS/INS estimate")
    ax.scatter(
        [f.p[1] for f in fixes], [f.p[0] for f in fixes], s=1.5, alpha=0.25, label="GNSS fixes"
    )
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Aided trajectory")
    fig.tight_layout()
    fig.savefig(outdir / "trajectory.png")
    plt.close(fig)

    # 2. error growth during one 120 s outage, per ablation
    snap = pass_result.snapshots[len(pass_result.snapshots) // 2]
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, cfg in outage_cfgs.items():
        errs, dists = _outage_trace(traj, imus, snap, cfg, 120.0)
        ax.plot(np.arange(len(errs)) * traj.dt, errs, lw=1.3, label=name)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("time since GNSS loss [s]")
    ax.set_ylabel("horizontal position error [m]")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title(f"Error growth during a 120 s outage (t0 = {snap.t:.0f} s)")
    fig.tight_layout()
    fig.savefig(outdir / "outage_error_growth.png")
    plt.close(fig)

    # 3. filter covariance vs actual error
    fig, ax = plt.subplots(figsize=(7, 3.5))
    err = np.linalg.norm(pass_result.p[:, :2] - traj.p[:, :2], axis=1)
    ax.plot(pass_result.t, err, lw=0.7, label="actual horizontal error")
    ax.plot(
        pass_result.t,
        3.0 * np.linalg.norm(pass_result.sigma_p[:, :2], axis=1),
        lw=1.0,
        ls="--",
        label="filter 3-sigma",
    )
    ax.set_xlabel("time [s]")
    ax.set_ylabel("[m]")
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)
    ax.set_title("Consistency: is the filter honest about its own uncertainty?")
    fig.tight_layout()
    fig.savefig(outdir / "covariance_consistency.png")
    plt.close(fig)
    print(f"\nplots written to {outdir}/")


def _outage_trace(traj, imus, snap, cfg, duration):
    """Per-epoch error trace through one outage, for plotting."""
    import copy

    from carcajou.pipeline import build_filter

    ekf = build_filter(traj, cfg)
    ekf.state, ekf.P = copy.deepcopy(snap.filter_state)
    n = int(round(duration / traj.dt))
    errs, dists = [], []
    for k in range(snap.index, min(snap.index + n, len(imus))):
        ekf.predict(imus[k], traj.dt)
        if cfg.use_zupt and ekf.is_stationary():
            ekf.update_zupt()
        if cfg.use_nhc and not ekf.is_stationary():
            ekf.update_nhc()
        e = ekf.state.p - traj.p[k + 1]
        errs.append(float(np.linalg.norm(e[:2])))
        dists.append(0.0)
    return np.array(errs), np.array(dists)


if __name__ == "__main__":
    raise SystemExit(main())
