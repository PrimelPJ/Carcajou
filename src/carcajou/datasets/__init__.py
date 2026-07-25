"""Dataset adapters and trajectory simulators.

``synthetic`` — algebraically self-consistent trajectory generator: IMU
measurements are derived by inverting the discrete mechanization update, so a
noise-free integration retraces the reference exactly (verified to machine
precision in ``tests/test_core.py::test_mechanization_reproduces_truth``).

``kitti`` — loader for the KITTI raw OXTS format with documented NED frame
conversions and a pre-flight validity check. No benchmark numbers in this
repository come from KITTI; the loader exists for real-data use.
"""

from .synthetic import (
    GnssFix,
    Segment,
    Trajectory,
    corrupt_imu,
    make_trajectory,
    perfect_imu,
    simulate_gnss,
)

__all__ = [
    "Trajectory",
    "Segment",
    "GnssFix",
    "make_trajectory",
    "perfect_imu",
    "corrupt_imu",
    "simulate_gnss",
]
