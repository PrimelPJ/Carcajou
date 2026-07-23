"""carcajou: a GNSS-denied navigation stack.

Phase 0 scope: strapdown INS mechanization, a 15-state error-state Kalman
filter with GNSS / ZUPT / NHC aiding, a self-consistent trajectory simulator,
a KITTI loader, and the GNSS-outage drift benchmark everything is judged by.
"""

from .eskf import Eskf, EskfConfig
from .frames import LocalTangentPlane
from .mechanization import ImuSample, Mechanizer, NavState
from .sensors import GNSS_GRADES, IMU_GRADES, GnssSpec, ImuSpec

__version__ = "0.1.0"
__all__ = [
    "Eskf",
    "EskfConfig",
    "ImuSample",
    "Mechanizer",
    "NavState",
    "LocalTangentPlane",
    "ImuSpec",
    "GnssSpec",
    "IMU_GRADES",
    "GNSS_GRADES",
]
