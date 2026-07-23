"""Sensor error models.

The IMU grades below are representative datasheet figures, not any one
vendor's part. They exist so the outage benchmark can answer the question that
actually matters commercially: *how cheap can the inertial sensor be and still
hold the drift budget when GNSS drops out?*
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEG = np.pi / 180.0
HR = 3600.0


@dataclass(frozen=True)
class ImuSpec:
    """Stochastic error model for a single IMU grade.

    Attributes
    ----------
    arw : angle random walk, rad/s/sqrt(Hz) (gyro white noise density)
    vrw : velocity random walk, m/s^2/sqrt(Hz) (accel white noise density)
    bi_g, bi_a : in-run bias instability, rad/s and m/s^2
    tau_g, tau_a : Gauss-Markov correlation times, s
    b0_g, b0_a : one-sigma turn-on bias, rad/s and m/s^2
    """

    name: str
    arw: float
    vrw: float
    bi_g: float
    bi_a: float
    tau_g: float
    tau_a: float
    b0_g: float
    b0_a: float

    def process_noise(self) -> np.ndarray:
        """Continuous-time process-noise PSD for ``[n_a, n_g, w_ba, w_bg]``."""
        # Gauss-Markov driving PSD that yields the specified steady-state sigma.
        q_ba = 2.0 * self.bi_a**2 / self.tau_a
        q_bg = 2.0 * self.bi_g**2 / self.tau_g
        return np.diag(
            np.concatenate(
                [
                    np.full(3, self.vrw**2),
                    np.full(3, self.arw**2),
                    np.full(3, q_ba),
                    np.full(3, q_bg),
                ]
            )
        )


# Consumer MEMS: the phone/dashcam class part. This is the hard case, and the
# one that makes the sensor-fusion argument, because it cannot hold 1% alone.
CONSUMER_MEMS = ImuSpec(
    name="consumer-mems",
    arw=0.30 * DEG / 60.0,  # 0.30 deg/sqrt(hr) -> rad/s/sqrt(Hz)
    vrw=0.10 / 60.0,  # 0.10 m/s/sqrt(hr) -> m/s^2/sqrt(Hz)
    bi_g=15.0 * DEG / HR,
    bi_a=0.5e-3 * 9.81,
    tau_g=300.0,
    tau_a=300.0,
    b0_g=200.0 * DEG / HR,
    b0_a=15.0e-3 * 9.81,
)

# Industrial / automotive MEMS, the grade Gulo-Gulo-class systems actually ship.
INDUSTRIAL_MEMS = ImuSpec(
    name="industrial-mems",
    arw=0.15 * DEG / 60.0,
    vrw=0.023 / 60.0,
    bi_g=2.0 * DEG / HR,
    bi_a=0.036e-3 * 9.81,
    tau_g=600.0,
    tau_a=600.0,
    b0_g=20.0 * DEG / HR,
    b0_a=2.0e-3 * 9.81,
)

# Tactical grade, for the upper bound on what money buys.
TACTICAL = ImuSpec(
    name="tactical",
    arw=0.02 * DEG / 60.0,
    vrw=0.008 / 60.0,
    bi_g=0.3 * DEG / HR,
    bi_a=0.010e-3 * 9.81,
    tau_g=3600.0,
    tau_a=3600.0,
    b0_g=1.0 * DEG / HR,
    b0_a=0.5e-3 * 9.81,
)

IMU_GRADES = {s.name: s for s in (CONSUMER_MEMS, INDUSTRIAL_MEMS, TACTICAL)}


@dataclass(frozen=True)
class GnssSpec:
    """GNSS receiver error model."""

    name: str
    sigma_horizontal: float  # m, 1-sigma
    sigma_vertical: float  # m, 1-sigma
    sigma_velocity: float  # m/s, 1-sigma (Doppler-derived)
    rate_hz: float = 10.0

    def R_position(self) -> np.ndarray:
        return np.diag(
            [self.sigma_horizontal**2, self.sigma_horizontal**2, self.sigma_vertical**2]
        )

    def R_velocity(self) -> np.ndarray:
        return np.eye(3) * self.sigma_velocity**2


SPP = GnssSpec("spp", 2.5, 4.5, 0.10)  # single-point, open sky
RTK = GnssSpec("rtk", 0.02, 0.03, 0.01)  # fixed-ambiguity RTK
URBAN = GnssSpec("urban", 8.0, 15.0, 0.35)  # multipath-degraded urban canyon

GNSS_GRADES = {s.name: s for s in (SPP, RTK, URBAN)}
