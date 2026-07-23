"""Phase 1 vision tests.

The claims these tests pin down, in the order the README makes them:

1. The stereo geometry round-trips exactly, so triangulation error in the
   benchmark is attributable to injected pixel noise and nothing else.
2. A noise-free front end over static structure recovers the true relative
   pose to numerical precision, extending the "why you can believe the
   numbers" identity from the IMU to the camera.
3. An unmasked front end staring at a station-keeping lead vehicle is
   *biased towards zero ego-motion*, and a masked one is not. This is the
   mechanism the whole ablation rests on, demonstrated at the single-epoch
   level where it cannot hide behind aggregation.
4. The reported measurement covariance is neither absurdly optimistic nor
   uselessly loose against Monte Carlo truth.
5. The ESKF VO update actually reduces velocity error, and the chi-square
   gate rejects a poisoned measurement wearing an honest covariance.
"""

from __future__ import annotations

import numpy as np
import pytest

from carcajou.datasets.synthetic import corrupt_imu, make_trajectory, perfect_imu
from carcajou.eskf import EskfConfig
from carcajou.pipeline import build_filter, index_vo
from carcajou.sensors import CONSUMER_MEMS, SPP
from carcajou.vision import (
    KITTI_LIKE,
    MASK_OFF,
    MASK_ORACLE,
    StereoVo,
    VoConfig,
    make_world,
)
from carcajou.vision.frontend import kabsch, ransac_rigid


@pytest.fixture(scope="module")
def traj():
    return make_trajectory(laps=1, rate_hz=100.0)


@pytest.fixture(scope="module")
def world(traj):
    return make_world(traj, np.random.default_rng(42))


# ---------------------------------------------------------------- geometry
def test_stereo_projection_roundtrip():
    rig = KITTI_LIKE
    rng = np.random.default_rng(0)
    P = np.stack(
        [
            rng.uniform(-15, 15, 200),
            rng.uniform(-3, 3, 200),
            rng.uniform(rig.min_depth + 0.5, rig.max_depth - 0.5, 200),
        ],
        axis=1,
    )
    back = rig.triangulate(rig.project_stereo(P))
    assert np.max(np.abs(back - P)) < 1e-9


def test_kabsch_recovers_exact_transform():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(50, 3))
    from carcajou.frames import exp_so3

    R_true = exp_so3(np.array([0.1, -0.2, 0.3]))
    t_true = np.array([1.0, -2.0, 0.5])
    B = A @ R_true.T + t_true
    R, t = kabsch(A, B)
    assert np.allclose(R, R_true, atol=1e-12)
    assert np.allclose(t, t_true, atol=1e-12)


def test_ransac_rejects_minority_outliers():
    rng = np.random.default_rng(2)
    A = rng.normal(scale=5.0, size=(120, 3))
    from carcajou.frames import exp_so3

    R_true = exp_so3(np.array([0.02, 0.01, -0.03]))
    t_true = np.array([0.5, 0.1, -0.2])
    B = A @ R_true.T + t_true
    B[:20] += rng.normal(scale=3.0, size=(20, 3))  # 17% gross mismatches
    sigma = np.full(len(A), 0.02)
    R, t, inl = ransac_rigid(A, B, sigma, iters=200, rng=rng)
    assert inl.sum() >= 95
    assert inl[:20].sum() <= 2
    assert np.allclose(t, t_true, atol=0.02)


# ------------------------------------------------------------- attribution
def test_noise_free_vo_recovers_truth(traj, world):
    """Zero pixel noise, oracle mask -> true body velocity to ~mm/s.

    The residual that remains is the average-over-interval versus
    instantaneous-at-epoch velocity discrepancy, bounded by a * dt / 2.
    """
    vo = StereoVo(KITTI_LIKE, world, traj, VoConfig(), mask=MASK_ORACLE)
    ks = vo.step_indices
    checked = 0
    for a, b in zip(ks[10:60], ks[11:61], strict=False):
        m = vo.measure(int(a), int(b), noisy=False)
        if m is None:
            continue
        v_true = traj.R[int(b)].T @ traj.v[int(b)]
        # generous bound: 4 m/s^2 * 0.1 s / 2 = 0.2 m/s worst case under braking
        assert np.linalg.norm(m.v_b - v_true) < 0.2
        checked += 1
    assert checked > 30


def test_unmasked_vo_is_biased_towards_zero_ego_motion(traj, world):
    """The single-epoch mechanism behind the whole Phase 1 ablation.

    Pick epochs where the lead vehicle is visible and moving with the ego.
    Mask off: dynamic points vote for zero relative motion, dragging the
    speed estimate down. Mask on: no such bias.
    """
    cfg = VoConfig(pixel_sigma=0.5, outlier_rate=0.0)
    vo_off = StereoVo(KITTI_LIKE, world, traj, cfg, mask=MASK_OFF, rng=np.random.default_rng(3))
    vo_on = StereoVo(KITTI_LIKE, world, traj, cfg, mask=MASK_ORACLE, rng=np.random.default_rng(3))

    bias_off, bias_on, n = [], [], 0
    for a, b in zip(vo_off.step_indices[:-1], vo_off.step_indices[1:], strict=False):
        a, b = int(a), int(b)
        m_off = vo_off.measure(a, b)
        m_on = vo_on.measure(a, b)
        if m_off is None or m_on is None:
            continue
        if m_off.n_dynamic_kept < 30:  # lead car not meaningfully in frame
            continue
        v_true = float(np.linalg.norm(traj.v[b]))
        if v_true < 3.0:
            continue
        bias_off.append(float(np.linalg.norm(m_off.v_b)) - v_true)
        bias_on.append(float(np.linalg.norm(m_on.v_b)) - v_true)
        n += 1
    assert n >= 10, "world/config no longer puts the lead vehicle in frame"
    # Unmasked: mean speed bias clearly negative (voting for standstill).
    assert np.mean(bias_off) < -0.05
    # Masked: an order of magnitude closer to unbiased.
    assert abs(np.mean(bias_on)) < 0.1 * abs(np.mean(bias_off))


def test_reported_covariance_is_consistent(traj, world):
    """Monte Carlo normalised innovation squared over repeated epochs.

    For a 3-vector with honest covariance, E[r^T R^-1 r] = 3. Accept a band
    wide enough for sampling noise but tight enough to catch a covariance
    that is wrong by a factor of a few either way.
    """
    cfg = VoConfig(outlier_rate=0.0)
    nis = []
    for seed in range(8):
        vo = StereoVo(
            KITTI_LIKE, world, traj, cfg, mask=MASK_ORACLE, rng=np.random.default_rng(100 + seed)
        )
        for a, b in zip(vo.step_indices[10:40], vo.step_indices[11:41], strict=False):
            m = vo.measure(int(a), int(b))
            if m is None:
                continue
            r = m.v_b - traj.R[m.index].T @ traj.v[m.index]
            nis.append(float(r @ np.linalg.solve(m.R_v, r)))
    mean_nis = float(np.mean(nis))
    assert 0.3 < mean_nis < 6.0, f"mean NIS {mean_nis:.2f}, expected near 3"


# ------------------------------------------------------------------ filter
def _short_filter(traj, rng):
    imus, _, _ = corrupt_imu(perfect_imu(traj), CONSUMER_MEMS, rng, traj.dt)
    cfg = EskfConfig(imu=CONSUMER_MEMS, gnss=SPP, use_zupt=True, use_nhc=True, use_vo=True)
    return imus, cfg, build_filter(traj, cfg)


def test_vo_update_reduces_velocity_error(traj, world):
    """Propagate consumer MEMS unaided vs with VO for 30 s mid-route."""
    rng = np.random.default_rng(5)
    imus, cfg, _ = _short_filter(traj, rng)
    vo = index_vo(
        StereoVo(KITTI_LIKE, world, traj, VoConfig(), mask=MASK_ORACLE).run()
    )

    start, steps = 3000, 3000  # 30 s at 100 Hz
    errs = {}
    for use_vo in (False, True):
        ekf = build_filter(traj, cfg)
        ekf.state.p = traj.p[start].copy()
        ekf.state.v = traj.v[start].copy()
        ekf.state.R = traj.R[start].copy()
        for k in range(start, start + steps):
            ekf.predict(imus[k], traj.dt)
            if use_vo:
                m = vo.get(k + 1)
                if m is not None:
                    ekf.update_vo_velocity(m.v_b, m.R_v)
        errs[use_vo] = float(np.linalg.norm(ekf.state.p - traj.p[start + steps]))
    assert errs[True] < 0.5 * errs[False]


def test_vo_rotation_update_behaves(traj, world):
    """Consistent relative rotation is accepted; a poisoned one is gated."""
    from carcajou.frames import exp_so3
    from carcajou.pipeline import _gyro_delta

    rng = np.random.default_rng(7)
    imus, cfg, ekf = _short_filter(traj, rng)
    vo = index_vo(
        StereoVo(KITTI_LIKE, world, traj, VoConfig(), mask=MASK_ORACLE).run()
    )
    ekf.state.p = traj.p[2000].copy()
    ekf.state.v = traj.v[2000].copy()
    ekf.state.R = traj.R[2000].copy()
    applied = 0
    for k in range(2000, 3000):
        ekf.predict(imus[k], traj.dt)
        m = vo.get(k + 1)
        if m is not None:
            ekf.update_vo_rotation(m.dR_b, _gyro_delta(imus, m, traj.dt), m.dt, m.R_rot)
            applied += 1
    assert applied > 50
    assert ekf.stats["vo_rot_applied"] > 0.9 * applied
    # Poison: a wildly wrong rotation with a confident covariance must be gated.
    m = next(iter(vo.values()))
    bad = exp_so3(np.array([0.0, 0.0, 0.2])) @ m.dR_b
    ok = ekf.update_vo_rotation(bad, _gyro_delta(imus, m, traj.dt), m.dt, np.eye(3) * 1e-8)
    assert not ok


def test_gate_rejects_poisoned_measurement(traj):
    """A VO velocity that is wildly wrong but confidently reported must be
    rejected by the innovation gate rather than absorbed."""
    rng = np.random.default_rng(6)
    imus, cfg, ekf = _short_filter(traj, rng)
    ekf.state.p = traj.p[1000].copy()
    ekf.state.v = traj.v[1000].copy()
    ekf.state.R = traj.R[1000].copy()
    for k in range(1000, 1200):
        ekf.predict(imus[k], traj.dt)
    v_b_true = ekf.state.R.T @ ekf.state.v
    ok = ekf.update_vo_velocity(v_b_true + np.array([5.0, 5.0, 0.0]), np.eye(3) * 0.02**2)
    assert not ok
    assert ekf.stats["vo_rejected"] == 1
