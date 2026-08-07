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


def test_extended_state_restores_covariance_consistency():
    """Phase 3: with scale factor injected and GM GNSS error on, the 15-state
    filter's 3-sigma claim collapses and the 24-state filter's holds."""
    import numpy as np

    from carcajou.datasets.synthetic import (
        corrupt_imu,
        make_trajectory,
        perfect_imu,
        simulate_gnss,
    )
    from carcajou.eskf import EskfConfig
    from carcajou.pipeline import run_aided_pass
    from carcajou.sensors import GNSS_GRADES, IMU_GRADES

    traj = make_trajectory(laps=1, rate_hz=100.0)
    clean = perfect_imu(traj)
    imu_g, gnss_g = IMU_GRADES["industrial-mems"], GNSS_GRADES["spp-gm"]
    inside = {}
    for ext in (False, True):
        rng = np.random.default_rng(100)
        imus, _, _ = corrupt_imu(clean, imu_g, rng, traj.dt, inject_scale=True)
        fixes = simulate_gnss(traj, gnss_g, rng)
        cfg = EskfConfig(imu=imu_g, gnss=gnss_g, use_zupt=True, use_nhc=True, extended_state=ext)
        pr = run_aided_pass(traj, imus, fixes, cfg)
        err = np.linalg.norm((pr.p - traj.p)[:, :2], axis=1)
        bound = 3.0 * np.linalg.norm(pr.sigma_p[:, :2], axis=1)
        inside[ext] = float(np.mean((err <= bound)[3000:]))
    assert inside[False] < 0.5, "15-state unexpectedly consistent; injection broken?"
    assert inside[True] > 0.9


# ---------------------------------------------------------- wheel speed aiding
def test_wss_reduces_velocity_error(short_traj):
    """Wheel speed aiding should bound forward velocity error during outage."""
    traj = short_traj
    rng = np.random.default_rng(20)
    clean = perfect_imu(traj)
    imus, _, _ = corrupt_imu(clean, INDUSTRIAL_MEMS, rng, traj.dt)
    cfg = EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP, use_wss=True, wss_sigma=0.3)
    ekf = Eskf(Mechanizer(traj.lat0, traj.h0), cfg, traj.initial_state())

    # Run without GNSS (outage), only wheel speed
    for k, imu in enumerate(imus[:2000]):
        ekf.predict(imu, traj.dt)
        # Feed true forward speed as wheel speed measurement
        v_fwd = float(traj.R[k + 1].T[0] @ traj.v[k + 1])
        v_fwd += rng.normal(0, 0.3)  # add noise matching sigma
        ekf.update_wss(v_fwd)

    assert ekf.stats["wss_applied"] > 1000
    # Forward velocity should be bounded
    v_b = ekf.state.R.T @ ekf.state.v
    v_b_true = traj.R[2000].T @ traj.v[2000]
    assert abs(v_b[0] - v_b_true[0]) < 2.0, "WSS should bound forward velocity"


# ------------------------------------------------------------- lever arm
def test_lever_arm_improves_gnss_position(short_traj):
    """With a lever arm configured, the GNSS update should correctly account
    for the IMU-to-antenna offset."""
    traj = short_traj
    rng = np.random.default_rng(30)
    clean = perfect_imu(traj)
    imus, _, _ = corrupt_imu(clean, INDUSTRIAL_MEMS, rng, traj.dt)

    lever_arm = np.array([1.5, 0.0, -0.5])  # antenna 1.5m forward, 0.5m up
    # Simulate GNSS fixes at the antenna location, not the IMU
    fixes = simulate_gnss(traj, SPP, rng)
    for fix in fixes:
        idx = int(round(fix.t / traj.dt))
        idx = min(idx, len(traj.R) - 1)
        fix.p = fix.p + traj.R[idx] @ lever_arm

    # Filter WITH lever arm compensation
    cfg_la = EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP, lever_arm=lever_arm)
    ekf_la = Eskf(Mechanizer(traj.lat0, traj.h0), cfg_la, traj.initial_state())
    fi = 0
    for imu in imus:
        ekf_la.predict(imu, traj.dt)
        while fi < len(fixes) and fixes[fi].t <= imu.t + 1e-9:
            ekf_la.update_gnss_position(fixes[fi].p)
            fi += 1
    err_la = np.linalg.norm(ekf_la.state.p - traj.p[-1])

    # Filter WITHOUT lever arm compensation (should be worse)
    cfg_no = EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP)
    ekf_no = Eskf(Mechanizer(traj.lat0, traj.h0), cfg_no, traj.initial_state())
    fi = 0
    for imu in imus:
        ekf_no.predict(imu, traj.dt)
        while fi < len(fixes) and fixes[fi].t <= imu.t + 1e-9:
            ekf_no.update_gnss_position(fixes[fi].p)
            fi += 1
    err_no = np.linalg.norm(ekf_no.state.p - traj.p[-1])

    assert err_la < err_no, (
        f"lever arm compensation should reduce error: {err_la:.3f} vs {err_no:.3f}"
    )


# --------------------------------------------------------- protection levels
def test_protection_levels_bound_error(short_traj):
    """Protection levels should bound the actual position error."""
    traj = short_traj
    rng = np.random.default_rng(40)
    clean = perfect_imu(traj)
    imus, _, _ = corrupt_imu(clean, INDUSTRIAL_MEMS, rng, traj.dt)
    fixes = simulate_gnss(traj, SPP, rng)
    cfg = EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP)
    ekf = Eskf(Mechanizer(traj.lat0, traj.h0), cfg, traj.initial_state())

    fi = 0
    for imu in imus:
        ekf.predict(imu, traj.dt)
        while fi < len(fixes) and fixes[fi].t <= imu.t + 1e-9:
            ekf.update_gnss_position(fixes[fi].p)
            if cfg.use_gnss_velocity:
                ekf.update_gnss_velocity(fixes[fi].v)
            fi += 1

    hpl, vpl = ekf.protection_levels(integrity_risk=1e-3)
    err_h = float(np.linalg.norm(ekf.state.p[:2] - traj.p[-1][:2]))
    err_v = float(abs(ekf.state.p[2] - traj.p[-1][2]))

    # PLs must be positive and should bound the error at this integrity risk
    assert hpl > 0.0
    assert vpl > 0.0
    assert err_h < hpl, f"horizontal error {err_h:.2f} exceeds HPL {hpl:.2f}"
    assert err_v < vpl, f"vertical error {err_v:.2f} exceeds VPL {vpl:.2f}"


def test_protection_levels_grow_during_outage(short_traj):
    """Protection levels should increase when GNSS is unavailable."""
    traj = short_traj
    rng = np.random.default_rng(41)
    clean = perfect_imu(traj)
    imus, _, _ = corrupt_imu(clean, INDUSTRIAL_MEMS, rng, traj.dt)
    fixes = simulate_gnss(traj, SPP, rng)
    cfg = EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP)
    ekf = Eskf(Mechanizer(traj.lat0, traj.h0), cfg, traj.initial_state())

    # Converge with GNSS for 2000 epochs
    fi = 0
    for imu in imus[:2000]:
        ekf.predict(imu, traj.dt)
        while fi < len(fixes) and fixes[fi].t <= imu.t + 1e-9:
            ekf.update_gnss_position(fixes[fi].p)
            ekf.update_gnss_velocity(fixes[fi].v)
            fi += 1

    hpl_before, _ = ekf.protection_levels()

    # Run 1000 more epochs without GNSS (outage)
    for imu in imus[2000:3000]:
        ekf.predict(imu, traj.dt)

    hpl_after, _ = ekf.protection_levels()
    assert hpl_after > hpl_before, "HPL should grow during GNSS outage"


def test_velocity_sigma_grows_during_outage(short_traj):
    """Velocity uncertainty should increase when GNSS is unavailable."""
    traj = short_traj
    rng = np.random.default_rng(42)
    clean = perfect_imu(traj)
    imus, _, _ = corrupt_imu(clean, INDUSTRIAL_MEMS, rng, traj.dt)
    fixes = simulate_gnss(traj, SPP, rng)
    cfg = EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP)
    ekf = Eskf(Mechanizer(traj.lat0, traj.h0), cfg, traj.initial_state())

    # Converge with GNSS for 2000 epochs
    fi = 0
    for imu in imus[:2000]:
        ekf.predict(imu, traj.dt)
        while fi < len(fixes) and fixes[fi].t <= imu.t + 1e-9:
            ekf.update_gnss_position(fixes[fi].p)
            ekf.update_gnss_velocity(fixes[fi].v)
            fi += 1

    vel_sig_before = ekf.velocity_sigma()

    # Run 1000 more epochs without GNSS (outage)
    for imu in imus[2000:3000]:
        ekf.predict(imu, traj.dt)

    vel_sig_after = ekf.velocity_sigma()
    assert float(np.linalg.norm(vel_sig_after)) > float(np.linalg.norm(vel_sig_before)), \
        "Velocity sigma should grow during GNSS outage"


def test_attitude_sigma_converges_with_gnss(short_traj):
    """Attitude uncertainty should decrease as GNSS updates are applied."""
    traj = short_traj
    rng = np.random.default_rng(50)
    clean = perfect_imu(traj)
    imus, _, _ = corrupt_imu(clean, INDUSTRIAL_MEMS, rng, traj.dt)
    fixes = simulate_gnss(traj, SPP, rng)
    cfg = EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP)
    ekf = Eskf(Mechanizer(traj.lat0, traj.h0), cfg, traj.initial_state())

    att_sig_initial = ekf.attitude_sigma()
    assert att_sig_initial[2] > att_sig_initial[0], "Yaw prior should be wider than roll/pitch"

    fi = 0
    for imu in imus[:3000]:
        ekf.predict(imu, traj.dt)
        while fi < len(fixes) and fixes[fi].t <= imu.t + 1e-9:
            ekf.update_gnss_position(fixes[fi].p)
            ekf.update_gnss_velocity(fixes[fi].v)
            fi += 1

    att_sig_after = ekf.attitude_sigma()
    # Yaw sigma should have converged from the 5-degree prior
    assert att_sig_after[2] < att_sig_initial[2], "Yaw sigma should decrease with GNSS aiding"


def test_bias_sigma_decreases_with_aiding(short_traj):
    """Bias uncertainty should shrink as the filter converges with GNSS."""
    traj = short_traj
    rng = np.random.default_rng(51)
    clean = perfect_imu(traj)
    imus, _, _ = corrupt_imu(clean, INDUSTRIAL_MEMS, rng, traj.dt)
    fixes = simulate_gnss(traj, SPP, rng)
    cfg = EskfConfig(imu=INDUSTRIAL_MEMS, gnss=SPP)
    ekf = Eskf(Mechanizer(traj.lat0, traj.h0), cfg, traj.initial_state())

    sig_ba_init, sig_bg_init = ekf.bias_sigma()
    assert sig_ba_init.shape == (3,)
    assert sig_bg_init.shape == (3,)

    fi = 0
    for imu in imus[:3000]:
        ekf.predict(imu, traj.dt)
        while fi < len(fixes) and fixes[fi].t <= imu.t + 1e-9:
            ekf.update_gnss_position(fixes[fi].p)
            ekf.update_gnss_velocity(fixes[fi].v)
            fi += 1

    sig_ba_after, sig_bg_after = ekf.bias_sigma()
    assert float(np.linalg.norm(sig_ba_after)) < float(np.linalg.norm(sig_ba_init)), \
        "Accel bias sigma should decrease with aiding"
    assert float(np.linalg.norm(sig_bg_after)) < float(np.linalg.norm(sig_bg_init)), \
        "Gyro bias sigma should decrease with aiding"


def test_body_velocity_property():
    """Body velocity should correctly transform nav-frame velocity."""
    from carcajou.mechanization import NavState

    # Vehicle heading 45 deg (NE), moving at 10 m/s in that direction
    yaw = np.pi / 4
    R = euler_to_dcm(0.0, 0.0, yaw)
    v_nav = np.array([10.0 / np.sqrt(2), 10.0 / np.sqrt(2), 0.0])

    state = NavState(v=v_nav, R=R)
    v_body = state.body_velocity()

    # In body frame: should be purely forward (~10 m/s)
    assert abs(v_body[0] - 10.0) < 1e-9, "Forward velocity should be 10 m/s"
    assert abs(v_body[1]) < 1e-9, "Right velocity should be zero"
    assert abs(v_body[2]) < 1e-9, "Down velocity should be zero"
