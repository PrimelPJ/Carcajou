"""carcajou: a GNSS-denied navigation stack.

Phases completed
----------------
0  Strapdown INS mechanization, 15-state ESKF with GNSS / ZUPT / NHC aiding,
   self-consistent trajectory simulator, KITTI loader, outage drift benchmark.
1  Stereo visual odometry (VO) as an ESKF measurement update; semantic
   segmentation mask removes dynamic-object correspondences before pose
   estimation; ablated mask-on / mask-off on the same benchmark harness.
2  LiDAR scan-to-map localization: map built on a GNSS-aided pass with dynamic
   returns removed, then used for bounded-error localization on a second pass.
3  24-state ESKF extension (accel/gyro scale factor + GNSS Gauss-Markov state)
   that closes the covariance-optimism defect; 15-state tables are unchanged.
"""

from .eskf import Eskf, EskfConfig
from .frames import LocalTangentPlane, dcm_to_quaternion
from .mechanization import ImuSample, Mechanizer, NavState
from .sensors import GNSS_GRADES, IMU_GRADES, GnssSpec, ImuSpec

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "Eskf",
    "EskfConfig",
    "ImuSample",
    "Mechanizer",
    "NavState",
    "LocalTangentPlane",
    "dcm_to_quaternion",
    "ImuSpec",
    "GnssSpec",
    "IMU_GRADES",
    "GNSS_GRADES",
]
