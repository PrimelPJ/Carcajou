"""Core correctness tests.

``test_mechanization_reproduces_truth`` is the load-bearing one. If the
noise-free INS does not retrace the reference trajectory to sub-millimetre
accuracy, every benchmark number downstream is meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from carcajou.datasets.synthetic import (
    Segment,
    corrupt_imu,
    make_trajectory,
    perfect_imu,
    simulate_gnss,
)
from carcajou.eskf import Eskf, EskfConfig
from carcajou.frames import (
    LocalTangentPlane,
    dcm_to_euler,
    euler_to_dcm,
    exp_so3,
    log_so3,
    skew,
)
from carcajou.mechanization import Mechanizer
from carcajou.sensors import INDUSTRIAL_MEMS, SPP


# --------------------------------------------------------------------- frames
def test_skew_matches_cross_product():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=3), rng.normal(size=3)
    assert np.allclose(skew(a) @ b, np.cross(a, b))


def test_exp_log_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        phi = rng.normal(size=3)
        phi *= rng.uniform(0.0, 3.0) / np.linalg.norm(phi)
        assert np.allclose(log_so3(exp_so3(phi)), phi, atol=1e-10)


def test_exp_so3_is_orthonormal_near_zero():
    R = exp_so3(np.array([1e-14, -2e-14, 3e-14]))
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)


def test_euler_roundtrip():
    rng = np.random.default_rng(2)
    for _ in range(50):
        roll = rng.uniform(-np.pi / 2 + 0.1, np.pi / 2 - 0.1)
        pitch = rng.uniform(-np.pi / 2 + 0.2, np.pi / 2 - 0.2)
        yaw = rng.uniform(-np.pi, np.pi)
        r2, p2, y2 = dcm_to_euler(euler_to_dcm(roll, pitch, yaw))
        assert np.allclose([r2, p2, y2], [roll, pitch, yaw], atol=1e-9)


def test_tangent_plane_roundtrip():
    ltp = LocalTangentPlane(np.deg2rad(51.0447), np.deg2rad(-114.0719), 1045.0)
    ned = np.array([1234.0, -567.0, 89.0])
    lat, lon, h = ltp.ned_to_llh(ned)
    assert np.allclose(ltp.llh_to_ned(lat, lon, h), ned, atol=1e-6)


# -------------------------------------------------------------- mechanization
@pytest.fixture(scope="module")
def short_traj():
    return make_trajectory(laps=1, rate_hz=100.0)


def test_mechanization_reproduces_truth(short_traj):
    """Noise-free INS must retrace the generated trajectory exactly."""
    traj = short_traj
    imus = perfect_imu(traj)
    mech = Mechanizer(traj.lat0, traj.h0)

    state = traj.initial_state()
    for imu in imus:
        state = mech.propagate(state, imu, traj.dt)

    pos_err = np.linalg.norm(state.p - traj.p[-1])
    vel_err = np.linalg.norm(state.v - traj.v[-1])
    att_err = np.rad2deg(np.linalg.norm(log_so3(state.R @ traj.R[-1].T)))

    assert pos_err < 1e-3, f"position closure {pos_err:.3e} m"
    assert vel_err < 1e-5, f"velocity closure {vel_err:.3e} m/s"
    assert att_err < 1e-6, f"attitude closure {att_err:.3e} deg"


def test_pure_ins_drifts_without_aiding(short_traj):
    """Sanity check in the other direction: a real IMU must drift."""
    traj = short_traj
    rng = np.random.default_rng(7)
    imus, _, _ = corrupt_imu(perfect_imu(traj), INDUSTRIAL_MEMS, rng, traj.dt)
    mech = Mechanizer(traj.lat0, traj.h0)

    state = traj.initial_state()
    for imu in imus:
        state = mech.propagate(state, imu, traj.dt)

    assert np.linalg.norm(state.p - traj.p[-1]) > 1.0


# ---------------------------------------------------------------------- eskf
def test_filter_estimates_imu_biases(short_traj):
    """With GNSS available the filter must observe the accelerometer and
    gyro biases, not just track position."""
    traj = short_traj
    rng = np.random.default_rng(11)
    clean = perfect_imu(traj)
    imus, b_a, b_g = corrupt_imu(clean, INDUSTRIAL_MEMS, rng, traj.dt)
    fixes = simulate_gnss(traj, SPP, rng)

    cfg = EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP)
    ekf = Eskf(Mechanizer(traj.lat0, traj.h0), cfg, traj.initial_state())

    fi = 0
    for imu in imus:
        ekf.predict(imu, traj.dt)
        while fi < len(fixes) and fixes[fi].t <= imu.t + 1e-9:
            ekf.update_gnss_position(fixes[fi].p)
            ekf.update_gnss_velocity(fixes[fi].v)
            fi += 1
        if ekf.is_stationary():
            ekf.update_zupt()

    assert np.linalg.norm(ekf.state.p - traj.p[-1]) < 3.0
    # Bias estimate must beat the turn-on prior it started from.
    assert np.linalg.norm(ekf.state.b_g - b_g[-1]) < np.linalg.norm(b_g[-1])
    assert np.linalg.norm(ekf.state.b_a - b_a[-1]) < np.linalg.norm(b_a[-1])


def test_covariance_stays_positive_definite(short_traj):
    traj = short_traj
    rng = np.random.default_rng(13)
    imus, _, _ = corrupt_imu(perfect_imu(traj), INDUSTRIAL_MEMS, rng, traj.dt)
    fixes = simulate_gnss(traj, SPP, rng)
    cfg = EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP)
    ekf = Eskf(Mechanizer(traj.lat0, traj.h0), cfg, traj.initial_state())

    fi = 0
    for imu in imus[:3000]:
        ekf.predict(imu, traj.dt)
        while fi < len(fixes) and fixes[fi].t <= imu.t + 1e-9:
            ekf.update_gnss_position(fixes[fi].p)
            fi += 1
        assert np.all(np.linalg.eigvalsh(ekf.P) > -1e-12)


def test_gate_rejects_gross_outlier(short_traj):
    traj = short_traj
    cfg = EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP)
    ekf = Eskf(Mechanizer(traj.lat0, traj.h0), cfg, traj.initial_state())
    assert ekf.update_gnss_position(traj.p[0] + np.array([500.0, 0.0, 0.0])) is False
    assert ekf.stats["gnss_rejected"] == 1


def test_zupt_detector_fires_when_stopped():
    traj = make_trajectory(laps=1, lap=[Segment(20.0, 0.0)], rate_hz=100.0)
    rng = np.random.default_rng(17)
    imus, _, _ = corrupt_imu(perfect_imu(traj), INDUSTRIAL_MEMS, rng, traj.dt)
    cfg = EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP)
    ekf = Eskf(Mechanizer(traj.lat0, traj.h0), cfg, traj.initial_state())
    for imu in imus:
        ekf.predict(imu, traj.dt)
        if ekf.is_stationary():
            ekf.update_zupt()
    assert ekf.stats["zupt"] > 100
