# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- Stochastic cloning for VO rotation (`clone_attitude`, `marginalize_clone`,
  `update_vo_rotation_cloned` in `Eskf`); the relative-rotation channel now
  properly tracks cross-epoch correlation and needs no covariance inflation
- `--extended-state` flag in `run_benchmark.py`
- Glossary table in README covering all abbreviated terms
- TL;DR section in README for non-specialist readers
- CONTRIBUTING.md, SECURITY.md, CITATION.cff
- GitHub issue templates and pull request template
- `OutageResult.summary_str()` for human-readable logging
- `err_m_mean` and `drift_pct_mean` fields in `summarize()` output
- OCI labels on the Docker image
- pip cache in CI matrix

### Changed
- Phase 3 roadmap marked done; C++/Eigen port, ROS2, and HIL replay moved to
  "Future infrastructure"
- `_update` and `_inject` now handle augmented P dimension from cloning
- Pipeline (`run_aided_pass`, `run_outage`) uses stochastic cloning when
  `use_vo_rotation` is enabled instead of the unclonable `update_vo_rotation`

### Fixed
- Author name in `pyproject.toml` and `LICENSE` corrected to Primel Jayawardana
- Clone URL in README Quickstart corrected to canonical `Carcajou` casing
- CI and Python/license badges restored to README header

---

## [0.3.0] — Phase 3: covariance-optimism fix

### Added
- 24-state ESKF extension (`EskfConfig.extended_state`): accel/gyro scale factor as random constants + first-order Gauss-Markov GNSS position error state
- `scripts/run_consistency.py` — 15- vs 24-state 3-sigma consistency check; asserted in CI
- `corrupt_imu(inject_scale=True)` and `spp-gm` receiver grade for the extended-state scenario

### Result
- 15-state filter: 0.3 % of windows inside own 3-sigma bound (a filter confidently lying)
- 24-state filter: 100 % — closes the covariance-optimism defect

---

## [0.2.0] — Phase 2: LiDAR scan-to-map

### Added
- `src/carcajou/lidar/` — scanner, ICP registration, voxel map, online matcher
- `scripts/run_lidar_ablation.py` — Phase 2 ablation table
- Two-pass benchmark: GNSS-aided map build → masked dynamic removal → second-pass localization

### Result
- 120 s outage: 1.21 % / 16 m (constraints only) → 0.01 % / 0.15 m (map-aided)
- Error bounded rather than growing; phantom-object analysis included

---

## [0.1.0] — Phase 1: stereo visual odometry

### Added
- `src/carcajou/vision/` — stereo rig, landmark world, simulated segmentation mask, VO front end
- `scripts/run_vision_ablation.py` — mask-on / mask-off ablation

### Result
- Consumer MEMS at 120 s: 1.28 % / 20 % → 0.42 % / 0.92 % with masked VO

---

## [0.0.1] — Phase 0: strapdown INS baseline

### Added
- `src/carcajou/` core: mechanization, 15-state ESKF, ZUPT, NHC, GNSS aiding
- Algebraically self-consistent trajectory simulator (closure to machine precision)
- KITTI raw OXTS loader
- `scripts/run_benchmark.py` — full ablation sweep
- 32-test suite; CI on Python 3.10 / 3.11 / 3.12
