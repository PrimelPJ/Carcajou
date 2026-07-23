"""KITTI raw dataset loader (OXTS packets).

Expects the standard raw layout::

    2011_09_30_drive_0033_sync/
      oxts/
        timestamps.txt
        data/0000000000.txt ...

Frame conventions
-----------------
KITTI's OXTS output is ENU/FLU flavoured; carcajou is NED/FRD. The conversions
applied here are:

===================  ==========================================
KITTI                carcajou
===================  ==========================================
``yaw`` (0 = east,   NED yaw (0 = north, CW+) = ``pi/2 - yaw``
CCW+)
``pitch`` (nose      NED pitch (nose up +) = ``-pitch``
down +)
``roll`` (left up +) NED roll (right down +) = ``roll``
``(af, al, au)``     body FRD = ``(af, -al, -au)``
``(wf, wl, wu)``     body FRD = ``(wf, -wl, -wu)``
===================  ==========================================

.. warning::
   Two things must be checked against a real sequence before you trust the
   output, and both are flagged by :func:`validate_against_truth`:

   1. **Gravity convention.** OXTS units differ in whether ``au`` includes
      gravity. carcajou's mechanization wants true specific force, which reads
      about ``-9.81`` on the FRD z-axis at rest. ``detect_gravity_convention``
      inspects a stationary span and reports what it found.
   2. **Rate.** The distributed raw sequences are 10 Hz. That is thin for
      strapdown integration; expect coning/sculling residuals that the
      synthetic benchmark does not exhibit. Prefer the 100 Hz unsynced
      ``extract`` packets where available.
"""

from __future__ import annotations

import datetime as _dt
import pathlib
from dataclasses import dataclass

import numpy as np

from ..frames import LocalTangentPlane, euler_to_dcm
from ..mechanization import ImuSample
from .synthetic import GnssFix, Trajectory

# Field order in an OXTS data file.
OXTS_FIELDS = (
    "lat lon alt roll pitch yaw vn ve vf vl vu ax ay az af al au "
    "wx wy wz wf wl wu pos_accuracy vel_accuracy navstat numsats posmode velmode orimode"
).split()


@dataclass
class KittiSequence:
    traj: Trajectory
    imus: list[ImuSample]
    fixes: list[GnssFix]
    pos_accuracy: np.ndarray


def _read_timestamps(path: pathlib.Path) -> np.ndarray:
    ts = []
    for line in path.read_text().strip().splitlines():
        # KITTI stamps have nanosecond precision; datetime tops out at micro.
        head, frac = line.strip().rsplit(".", 1)
        base = _dt.datetime.strptime(head, "%Y-%m-%d %H:%M:%S")
        ts.append(base.timestamp() + float("0." + frac))
    t = np.asarray(ts, float)
    return t - t[0]


def _read_oxts(dirpath: pathlib.Path) -> np.ndarray:
    files = sorted(dirpath.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no OXTS packets under {dirpath}")
    return np.array([np.fromstring(f.read_text(), sep=" ") for f in files])


def detect_gravity_convention(f_frd: np.ndarray, still: slice) -> str:
    """Report whether a stationary span looks like specific force or free acceleration."""
    mean_z = float(np.mean(f_frd[still, 2]))
    if mean_z < -8.0:
        return "specific-force"  # what carcajou wants
    if abs(mean_z) < 2.0:
        return "gravity-removed"  # add gravity back before use
    return "unknown"


def load(
    sequence_dir: str | pathlib.Path,
    add_gravity_back: bool = False,
) -> KittiSequence:
    """Load one KITTI raw drive into carcajou's frames.

    Parameters
    ----------
    add_gravity_back
        Set when :func:`detect_gravity_convention` reports ``gravity-removed``.
    """
    root = pathlib.Path(sequence_dir)
    oxts_dir = root / "oxts"
    t = _read_timestamps(oxts_dir / "timestamps.txt")
    d = _read_oxts(oxts_dir / "data")
    col = {name: i for i, name in enumerate(OXTS_FIELDS)}

    lat = np.deg2rad(d[:, col["lat"]])
    lon = np.deg2rad(d[:, col["lon"]])
    alt = d[:, col["alt"]]

    ltp = LocalTangentPlane(float(lat[0]), float(lon[0]), float(alt[0]))
    p = np.array([ltp.llh_to_ned(la, lo, al) for la, lo, al in zip(lat, lon, alt, strict=True)])

    # ENU velocity (vn, ve, vu) -> NED
    v = np.stack([d[:, col["vn"]], d[:, col["ve"]], -d[:, col["vu"]]], axis=1)

    roll = d[:, col["roll"]]
    pitch = -d[:, col["pitch"]]
    yaw = np.pi / 2.0 - d[:, col["yaw"]]
    R = np.stack([euler_to_dcm(roll[i], pitch[i], yaw[i]) for i in range(len(t))], axis=0)

    f = np.stack([d[:, col["af"]], -d[:, col["al"]], -d[:, col["au"]]], axis=1)
    w = np.stack([d[:, col["wf"]], -d[:, col["wl"]], -d[:, col["wu"]]], axis=1)

    if add_gravity_back:
        from ..frames import gravity_ned

        g = gravity_ned(float(lat[0]), float(alt[0]))
        f = f - np.einsum("nji,j->ni", R, g)  # f_b -= R^T g

    traj = Trajectory(
        t=t, p=p, v=v, R=R, lat0=float(lat[0]), lon0=float(lon[0]), h0=float(alt[0])
    )
    imus = [ImuSample(t=float(t[k]), f=f[k], w=w[k]) for k in range(1, len(t))]
    fixes = [
        GnssFix(t=float(t[k]), p=p[k], v=v[k]) for k in range(len(t))
    ]  # OXTS is already a fused solution; treat as a reference-grade fix
    return KittiSequence(traj=traj, imus=imus, fixes=fixes, pos_accuracy=d[:, col["pos_accuracy"]])


def validate_against_truth(seq: KittiSequence, still: slice = slice(0, 20)) -> dict:
    """Cheap pre-flight checks. Run this before trusting any KITTI numbers."""
    f = np.array([s.f for s in seq.imus])
    dt = np.diff(seq.traj.t)
    return {
        "n_epochs": len(seq.traj.t),
        "rate_hz": float(1.0 / np.median(dt)),
        "dt_jitter_ms": float(np.std(dt) * 1e3),
        "gravity_convention": detect_gravity_convention(f, still),
        "mean_f_z_at_start": float(np.mean(f[still, 2])),
        "distance_m": float(seq.traj.arc_length()[-1]),
        "duration_s": float(seq.traj.t[-1]),
    }
