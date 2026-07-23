"""Tests for the outage harness and the metrics it reports.

Kept deliberately small and low-rate so CI stays fast. The full benchmark is a
separate job.
"""

from __future__ import annotations

import numpy as np
import pytest

from carcajou.benchmark.metrics import OutageResult, ate_rmse, summarize
from carcajou.datasets.synthetic import (
    corrupt_imu,
    make_trajectory,
    perfect_imu,
    simulate_gnss,
)
from carcajou.eskf import EskfConfig
from carcajou.pipeline import run_aided_pass, run_outage_study
from carcajou.sensors import INDUSTRIAL_MEMS, SPP

DURATIONS = [5.0, 20.0]


@pytest.fixture(scope="module")
def scenario():
    traj = make_trajectory(laps=1, rate_hz=50.0)
    rng = np.random.default_rng(42)
    imus, _, _ = corrupt_imu(perfect_imu(traj), INDUSTRIAL_MEMS, rng, traj.dt)
    fixes = simulate_gnss(traj, SPP, rng)
    return traj, imus, fixes


def test_aided_pass_beats_raw_gnss(scenario):
    """The filter must do better than the fixes it is fed, or it is pointless."""
    traj, imus, fixes = scenario
    cfg = EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP)
    res = run_aided_pass(traj, imus, fixes, cfg)

    filtered = ate_rmse(res.p[:, :2], traj.p[:, :2])
    idx = [int(round(f.t / traj.dt)) for f in fixes]
    raw = ate_rmse(np.array([f.p[:2] for f in fixes]), traj.p[idx][:, :2])

    assert filtered < raw
    assert filtered < SPP.sigma_horizontal


def test_filter_covariance_is_broadly_consistent(scenario):
    """The reported covariance must be in the right ballpark, in both directions.

    A filter that lies about its own uncertainty is worse than useless to
    whatever consumes it downstream. Measured behaviour: the horizontal
    channels sit around 95 to 98 percent inside 3-sigma rather than the
    nominal 99.7, i.e. mildly optimistic. That is expected and documented in
    docs/DESIGN.md section 9: GNSS error is modelled as white when real
    multipath is time-correlated, and IMU scale-factor and axis-misalignment
    errors are not in the state vector at all. The bound here catches gross
    overconfidence and gross conservatism, which are bugs; it does not pretend
    the residual optimism is absent.
    """
    traj, imus, fixes = scenario
    cfg = EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP)
    res = run_aided_pass(traj, imus, fixes, cfg)

    err = np.abs(res.p - traj.p)[500:]
    envelope = 3.0 * res.sigma_p[500:]
    per_axis = np.mean(err <= envelope, axis=0)
    assert np.all(per_axis > 0.93), f"overconfident, inside-3-sigma per axis = {per_axis}"

    # And not absurdly conservative either: a filter can always be "consistent"
    # by reporting a kilometre of uncertainty.
    ratio = np.median(envelope, axis=0) / np.maximum(np.percentile(err, 95, axis=0), 1e-6)
    assert np.all(ratio < 5.0), f"underconfident, 3-sigma / err_p95 = {ratio}"


def test_outage_drift_grows_with_duration(scenario):
    traj, imus, fixes = scenario
    cfg = EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP)
    _, study = run_outage_study(
        traj,
        imus,
        fixes,
        cfg,
        {"ins-only": EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP, use_zupt=False, use_nhc=False)},
        DURATIONS,
        window_spacing=30.0,
        warmup=30.0,
    )
    short = np.median([r.final_error_horizontal for r in study["ins-only"][5.0]])
    long = np.median([r.final_error_horizontal for r in study["ins-only"][20.0]])
    assert long > short


def test_constraints_reduce_outage_drift(scenario):
    """The headline claim of the whole project, asserted in CI."""
    traj, imus, fixes = scenario
    aided = EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP, use_zupt=True, use_nhc=True)
    cfgs = {
        "ins-only": EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP, use_zupt=False, use_nhc=False),
        "ins+zupt+nhc": EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP, use_zupt=True, use_nhc=True),
    }
    _, study = run_outage_study(
        traj, imus, fixes, aided, cfgs, DURATIONS, window_spacing=30.0, warmup=30.0
    )
    bare = summarize(study["ins-only"][20.0])
    aided_s = summarize(study["ins+zupt+nhc"][20.0])
    assert aided_s["drift_pct_median"] < bare["drift_pct_median"]


def test_summarize_shape_and_nan_handling():
    res = [
        OutageResult(0.0, 30.0, 300.0, 3.0, 2.0, 1.0, 2.5, 0.4),
        OutageResult(60.0, 30.0, 0.0, 3.0, 2.0, 1.0, 2.5, 0.4),  # stationary, drift is nan
    ]
    s = summarize(res)
    assert s["n_windows"] == 2
    assert np.isfinite(s["drift_pct_median"])
    assert summarize([]) == {}


def test_short_trajectory_raises_actionable_error(scenario):
    """Regression: CI once asked for a 120 s outage on a 224 s trajectory and
    got a bare 'too short' message. The error must say what to change."""
    traj, imus, fixes = scenario
    cfg = EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP)
    with pytest.raises(ValueError, match=r"--laps|--durations|--warmup"):
        run_outage_study(
            traj, imus, fixes, cfg, {"ins-only": cfg}, [1e4], window_spacing=30.0, warmup=30.0
        )
