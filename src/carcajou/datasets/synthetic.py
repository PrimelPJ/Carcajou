"""Self-consistent synthetic trajectory generator.

The point of this module is falsifiability. It generates truth *by inverting
the exact discrete mechanization equations in* :mod:`carcajou.mechanization`,
so a noise-free, bias-free run of the INS reproduces the reference trajectory
to machine precision. If ``test_mechanization.py`` ever fails, the navigation
core is wrong, not the tuning.

Once that identity holds, every metre of drift you see in the benchmark is
attributable to a sensor error you deliberately injected, and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frames import (
    earth_rate_ned,
    euler_to_dcm,
    gravity_ned,
    log_so3,
)
from ..mechanization import ImuSample, NavState
from ..sensors import GnssSpec, ImuSpec


@dataclass
class Trajectory:
    """Ground truth sampled at a fixed rate."""

    t: np.ndarray
    p: np.ndarray  # (N,3) NED position
    v: np.ndarray  # (N,3) NED velocity
    R: np.ndarray  # (N,3,3) body -> nav
    lat0: float
    lon0: float
    h0: float

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    def arc_length(self) -> np.ndarray:
        """Cumulative distance travelled, metres."""
        step = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(step)])

    def initial_state(self) -> NavState:
        return NavState(
            t=float(self.t[0]),
            p=self.p[0].copy(),
            v=self.v[0].copy(),
            R=self.R[0].copy(),
        )


@dataclass
class Segment:
    """One leg of a route. Speed ramps linearly to ``speed``; yaw turns by ``turn``."""

    duration: float
    speed: float
    turn: float = 0.0  # total heading change over the segment, radians


URBAN_LAP: list[Segment] = [
    Segment(8.0, 14.0),  # pull away from the light
    Segment(40.0, 14.0),  # straight block
    Segment(6.0, 10.0, turn=np.pi / 2),  # right turn
    Segment(40.0, 14.0),  # straight block
    Segment(6.0, 0.0),  # brake
    Segment(12.0, 0.0),  # stopped at the light (ZUPT opportunity)
]


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x
    k = np.ones(width) / width
    return np.convolve(np.pad(x, (width, width), mode="edge"), k, mode="same")[width:-width]


def make_trajectory(
    laps: int = 5,
    lap: list[Segment] | None = None,
    rate_hz: float = 100.0,
    lat0_deg: float = 51.0447,  # Calgary
    lon0_deg: float = -114.0719,
    h0: float = 1045.0,
    grade_amplitude: float = 8.0,
    grade_period: float = 220.0,
) -> Trajectory:
    """Build a driveable urban circuit as ground truth."""
    lap = lap or URBAN_LAP
    dt = 1.0 / rate_hz

    speed_cmd: list[float] = []
    yawrate_cmd: list[float] = []
    speed_prev = 0.0
    for _ in range(laps):
        for seg in lap:
            n = max(1, int(round(seg.duration / dt)))
            speed_cmd.extend(np.linspace(speed_prev, seg.speed, n))
            yawrate_cmd.extend([seg.turn / seg.duration] * n)
            speed_prev = seg.speed

    # Vehicles have finite jerk; smoothing the command keeps the derived
    # specific force free of the step discontinuities a real IMU never sees.
    w = max(1, int(round(0.8 * rate_hz)))
    speed = np.clip(_smooth(np.asarray(speed_cmd), w), 0.0, None)
    yawrate = _smooth(np.asarray(yawrate_cmd), w)

    n = len(speed)
    t = np.arange(n) * dt
    yaw = np.cumsum(yawrate) * dt

    # Gentle rolling grade so the vertical channel is genuinely exercised.
    v_down = -grade_amplitude * (2 * np.pi / grade_period) * np.cos(2 * np.pi * t / grade_period)
    v_down = np.where(speed > 0.1, v_down, 0.0)

    v = np.stack([speed * np.cos(yaw), speed * np.sin(yaw), v_down], axis=1)

    v_h = np.hypot(v[:, 0], v[:, 1])
    pitch = np.arctan2(-v[:, 2], np.maximum(v_h, 1e-6))
    pitch = np.where(v_h > 0.1, pitch, 0.0)
    # Suspension roll into the turn, ~15 percent of the coordinated-turn angle.
    g_mag = float(np.linalg.norm(gravity_ned(np.deg2rad(lat0_deg))))
    roll = 0.15 * np.arctan2(speed * yawrate, g_mag)

    R = np.stack([euler_to_dcm(roll[i], pitch[i], yaw[i]) for i in range(n)], axis=0)

    # Position by the same trapezoidal rule the mechanizer uses.
    p = np.zeros((n, 3))
    for k in range(n - 1):
        p[k + 1] = p[k] + 0.5 * (v[k] + v[k + 1]) * dt

    return Trajectory(
        t=t, p=p, v=v, R=R, lat0=np.deg2rad(lat0_deg), lon0=np.deg2rad(lon0_deg), h0=h0
    )


def perfect_imu(traj: Trajectory) -> list[ImuSample]:
    """Invert the mechanization to recover the exact IMU that produced ``traj``.

    Uses the discrete update in :meth:`Mechanizer.propagate`, not its continuous
    idealisation, so integrating these samples returns the trajectory exactly.
    """
    dt = traj.dt
    g = gravity_ned(traj.lat0, traj.h0)
    w_ie = earth_rate_ned(traj.lat0)
    n = len(traj.t)
    out: list[ImuSample] = []

    for k in range(n - 1):
        Rk, Rk1 = traj.R[k], traj.R[k + 1]

        # Angular rate: R_{k+1} = R_k Exp(w_nb_b dt)
        w_nb_b = log_so3(Rk.T @ Rk1) / dt
        w_ib_b = w_nb_b + Rk.T @ w_ie

        # Specific force: v_{k+1} = v_k + (f_n + g - 2 w_ie x v_k) dt
        # with f_n = 0.5 (R_k + R_{k+1}) f_b
        f_n = (traj.v[k + 1] - traj.v[k]) / dt - g + 2.0 * np.cross(w_ie, traj.v[k])
        M = 0.5 * (Rk + Rk1)
        f_b = np.linalg.solve(M, f_n)

        out.append(ImuSample(t=float(traj.t[k + 1]), f=f_b, w=w_ib_b))
    return out


def corrupt_imu(
    clean: list[ImuSample], spec: ImuSpec, rng: np.random.Generator, dt: float
) -> tuple[list[ImuSample], np.ndarray, np.ndarray]:
    """Add turn-on bias, Gauss-Markov in-run bias and white noise.

    Returns the corrupted samples plus the true bias histories, so the
    benchmark can score bias estimation as well as position.
    """
    n = len(clean)
    b_a = np.zeros((n, 3))
    b_g = np.zeros((n, 3))

    b_a_cur = rng.normal(0.0, spec.b0_a, 3) + rng.normal(0.0, spec.bi_a, 3)
    b_g_cur = rng.normal(0.0, spec.b0_g, 3) + rng.normal(0.0, spec.bi_g, 3)

    beta_a = np.exp(-dt / spec.tau_a)
    beta_g = np.exp(-dt / spec.tau_g)
    q_a = spec.bi_a * np.sqrt(1.0 - beta_a**2)
    q_g = spec.bi_g * np.sqrt(1.0 - beta_g**2)

    # Continuous PSD -> discrete sample standard deviation.
    sigma_f = spec.vrw / np.sqrt(dt)
    sigma_w = spec.arw / np.sqrt(dt)

    out: list[ImuSample] = []
    for k, s in enumerate(clean):
        b_a_cur = beta_a * b_a_cur + rng.normal(0.0, q_a, 3)
        b_g_cur = beta_g * b_g_cur + rng.normal(0.0, q_g, 3)
        b_a[k], b_g[k] = b_a_cur, b_g_cur
        out.append(
            ImuSample(
                t=s.t,
                f=s.f + b_a_cur + rng.normal(0.0, sigma_f, 3),
                w=s.w + b_g_cur + rng.normal(0.0, sigma_w, 3),
            )
        )
    return out, b_a, b_g


@dataclass
class GnssFix:
    t: float
    p: np.ndarray
    v: np.ndarray


def simulate_gnss(
    traj: Trajectory, spec: GnssSpec, rng: np.random.Generator
) -> list[GnssFix]:
    """Sample the trajectory at the receiver rate and add correlated-ish noise."""
    step = max(1, int(round((1.0 / spec.rate_hz) / traj.dt)))
    fixes: list[GnssFix] = []
    sigma_p = np.array([spec.sigma_horizontal, spec.sigma_horizontal, spec.sigma_vertical])
    for k in range(0, len(traj.t), step):
        fixes.append(
            GnssFix(
                t=float(traj.t[k]),
                p=traj.p[k] + rng.normal(0.0, sigma_p),
                v=traj.v[k] + rng.normal(0.0, spec.sigma_velocity, 3),
            )
        )
    return fixes
