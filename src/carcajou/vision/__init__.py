"""Phase 1: vision aiding.

Stereo visual odometry as an ESKF measurement update, with semantic
segmentation masking dynamic objects *before* feature correspondences reach the
pose estimator.

The headline ablation is mask on versus mask off on the same outage harness the
rest of the repository is scored by. Nothing else changes: same trajectory,
same IMU realisation, same snapshots, same metric.
"""

from .camera import KITTI_LIKE, NARROW_BASELINE, StereoRig
from .frontend import StereoVo, VoConfig, VoMeasurement
from .segmentation import (
    MASK_OFF,
    MASK_ORACLE,
    MASK_REALISTIC,
    OnnxSemanticMask,
    SimulatedMask,
)
from .world import LandmarkWorld, make_world

__all__ = [
    "StereoRig",
    "KITTI_LIKE",
    "NARROW_BASELINE",
    "StereoVo",
    "VoConfig",
    "VoMeasurement",
    "SimulatedMask",
    "OnnxSemanticMask",
    "MASK_OFF",
    "MASK_ORACLE",
    "MASK_REALISTIC",
    "LandmarkWorld",
    "make_world",
]
