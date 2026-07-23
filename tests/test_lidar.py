"""Phase 2 tests, in the order the claims are made:

1. A noise-free scan registered against a truth-posed map recovers the
   sensor pose to numerical precision — the attribution identity, extended
   from the IMU (Phase 0) and the camera (Phase 1) to the LiDAR.
2. The acceptance gates separate correct registrations from ICP that snapped
   to the wrong local minimum, so lost-in-map degrades to coasting instead
   of poisoning the filter.
3. The map file round-trips, and a version mismatch refuses to load rather
   than silently misinterpreting bytes.
4. Masking at map-build time keeps the dynamic actor out of the map; without
   it, a phantom vehicle is baked into the prior.
5. The map-pose update bounds filter error during an outage.
"""

from __future__ import annotations

import numpy as np
import pytest

from carcajou.datasets.synthetic import corrupt_imu, make_trajectory, perfect_imu
from carcajou.eskf import EskfConfig
from carcajou.frames import exp_so3, log_so3
from carcajou.lidar import LidarScanner, LidarSpec, build_map
from carcajou.lidar.mapping import LandmarkMap
from carcajou.lidar.matcher import MapMatcher
from carcajou.lidar.registration import icp_point_to_point
from carcajou.pipeline import build_filter
from carcajou.sensors import CONSUMER_MEMS, SPP
from carcajou.vision import MASK_OFF, MASK_ORACLE, make_world


@pytest.fixture(scope="module")
def traj():
    return make_trajectory(laps=1, rate_hz=100.0)


@pytest.fixture(scope="module")
def world(traj):
    return make_world(traj, np.random.default_rng(42), per_station=8)


@pytest.fixture(scope="module")
def scanner(traj, world):
    return LidarScanner(LidarSpec(), world, traj, rng=np.random.default_rng(7))


def _truth_poses(scanner, traj):
    p_bl = scanner.spec.p_bl
    return {
        int(k): (traj.R[int(k)], traj.p[int(k)] + traj.R[int(k)] @ p_bl)
        for k in scanner.step_indices
    }


@pytest.fixture(scope="module")
def truth_map(scanner, traj):
    scans = [scanner.scan(int(k), noisy=False) for k in scanner.step_indices[::3]]
    poses = _truth_poses(scanner, traj)
    return build_map(scans, poses, MASK_ORACLE, np.random.default_rng(0), voxel=0.4)


# ------------------------------------------------------------- attribution
def test_noise_free_registration_is_exact(traj, scanner, truth_map):
    tree = truth_map.tree()
    k = int(scanner.step_indices[45])
    s = scanner.scan(k, noisy=False)
    R_t = traj.R[k]
    t_t = traj.p[k] + R_t @ scanner.spec.p_bl
    res = icp_point_to_point(
        s.points, s.sigma, tree, truth_map.points, truth_map.sigma,
        exp_so3(np.deg2rad([0.5, -0.5, 2.0])) @ R_t, t_t + np.array([1.0, -1.0, 0.3]),
    )
    assert res.converged
    assert np.linalg.norm(res.t - t_t) < 1e-9
    assert np.linalg.norm(log_so3(res.R @ R_t.T)) < 1e-9


def test_wrong_basin_is_rejected_not_trusted(traj, world, scanner):
    """From far outside the basin, ICP snaps to a wrong alignment. The
    matcher's rms and match-fraction gates must refuse it."""
    poses = _truth_poses(scanner, traj)
    scans = [scanner.scan(int(k), noisy=True) for k in scanner.step_indices[::3]]
    lmap = build_map(scans, poses, MASK_ORACLE, np.random.default_rng(1), voxel=0.5)
    matcher = MapMatcher(scanner, lmap, mask=MASK_OFF, rng=np.random.default_rng(2))

    k = int(scanner.step_indices[50])
    R_t = traj.R[k]
    p_t = traj.p[k]
    good = matcher.measure(k, R_t, p_t + np.array([1.0, -0.5, 0.1]))
    assert good is not None
    assert np.linalg.norm(good.p_b - p_t) < 0.5

    rejected, accepted_far = 0, []
    for off in (8.0, 12.0, 16.0, 20.0):
        bad = matcher.measure(k, R_t, p_t + np.array([off, off * 0.5, 0.0]))
        if bad is None:
            rejected += 1
        else:
            accepted_far.append(np.linalg.norm(bad.p_b - p_t))
    # Far starts must overwhelmingly be refused; anything accepted must have
    # actually recovered the true pose, never a confident wrong one.
    assert rejected >= 3
    assert all(e < 0.5 for e in accepted_far)


# ------------------------------------------------------------- persistence
def test_map_roundtrip_and_version_guard(tmp_path, truth_map):
    path = tmp_path / "map.npz"
    truth_map.save(path)
    loaded = LandmarkMap.load(path)
    assert np.allclose(loaded.points, truth_map.points)
    assert np.allclose(loaded.sigma, truth_map.sigma)
    assert loaded.voxel == truth_map.voxel

    np.savez_compressed(
        tmp_path / "bad.npz", version=99, points=truth_map.points,
        sigma=truth_map.sigma, voxel=truth_map.voxel,
    )
    with pytest.raises(ValueError, match="rebuild"):
        LandmarkMap.load(tmp_path / "bad.npz")


def test_mask_keeps_phantom_vehicle_out_of_map(traj, world, scanner):
    """Build the same map with and without dynamic removal and count map
    points near the lead vehicle's swept positions."""
    poses = _truth_poses(scanner, traj)
    ks = scanner.step_indices[10:40]
    scans = [scanner.scan(int(k), noisy=True) for k in ks]

    counts = {}
    for name, mask in [("masked", MASK_ORACLE), ("unmasked", MASK_OFF)]:
        lmap = build_map(scans, poses, mask, np.random.default_rng(3), voxel=0.5)
        tree = lmap.tree()
        n_phantom = 0
        for k in ks[::4]:
            actor = world.actor_points(int(k))
            if len(actor):
                d, _ = tree.query(actor, k=1)
                n_phantom += int((d < 0.5).sum())
        counts[name] = n_phantom
    assert counts["unmasked"] > 20 * max(counts["masked"], 1)


# ------------------------------------------------------------------ filter
def test_map_update_bounds_outage_error(traj, world, scanner):
    poses = _truth_poses(scanner, traj)
    scans = [scanner.scan(int(k), noisy=True) for k in scanner.step_indices]
    lmap = build_map(scans, poses, MASK_ORACLE, np.random.default_rng(4), voxel=0.5)
    matcher = MapMatcher(
        LidarScanner(LidarSpec(), world, traj, rng=np.random.default_rng(9)),
        lmap, mask=MASK_OFF, rng=np.random.default_rng(5),
    )

    rng = np.random.default_rng(6)
    imus, _, _ = corrupt_imu(perfect_imu(traj), CONSUMER_MEMS, rng, traj.dt)
    cfg = EskfConfig(imu=CONSUMER_MEMS, gnss=SPP, use_zupt=True, use_nhc=True, use_map=True)

    start, steps = 2000, 6000  # 60 s at 100 Hz
    errs = {}
    for use_map in (False, True):
        ekf = build_filter(traj, cfg)
        ekf.state.p = traj.p[start].copy()
        ekf.state.v = traj.v[start].copy()
        ekf.state.R = traj.R[start].copy()
        for k in range(start, start + steps):
            ekf.predict(imus[k], traj.dt)
            if use_map and matcher.wants(k + 1):
                mm = matcher.measure(k + 1, ekf.state.R, ekf.state.p)
                if mm is not None:
                    ekf.update_map_position(mm.p_b, mm.cov_p)
                    ekf.update_map_rotation(mm.R_b, mm.cov_r)
        errs[use_map] = float(np.linalg.norm(ekf.state.p - traj.p[start + steps]))
    assert errs[True] < 0.3 * errs[False]
    assert errs[True] < 2.0  # bounded, not merely reduced
