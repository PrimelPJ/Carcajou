"""Phase 2: LiDAR scan matching against a persisted map.

Two-pass structure: map once with GNSS available and dynamic returns removed,
persist the static structure, then localize against it forever after. The
scored claim is accuracy recovery on the second pass over the same route —
same sensors, better answer, because prior structure is information.
"""

from .mapping import LandmarkMap, build_map
from .registration import RegistrationResult, icp_point_to_point, voxel_downsample
from .scanner import LidarScanner, LidarSpec, Scan

__all__ = [
    "LidarScanner",
    "LidarSpec",
    "Scan",
    "LandmarkMap",
    "build_map",
    "RegistrationResult",
    "icp_point_to_point",
    "voxel_downsample",
]
