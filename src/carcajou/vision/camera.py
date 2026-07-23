"""Stereo rig geometry.

Frames
------
Camera frame  : OpenCV convention, ``x`` right, ``y`` down, ``z`` forward.
Body frame    : FRD, as everywhere else in carcajou.
``R_bc``      : maps camera vectors into body, ``v_b = R_bc @ v_c``. Forward in
                camera is ``+z``, forward in body is ``+x``, so this is a fixed
                axis permutation and not a free parameter.

The rig is rectified by construction: both cameras share intrinsics and the
right camera is displaced along ``+x_c`` by the baseline. Real rectification is
an image-processing step; its output is exactly this model, so nothing
downstream of here changes when real images are substituted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Camera -> body axis permutation: x_c(right)->y_b, y_c(down)->z_b, z_c(fwd)->x_b
R_BC = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


@dataclass(frozen=True)
class StereoRig:
    """Rectified stereo pair rigidly mounted to the vehicle body.

    Attributes
    ----------
    fx, fy : focal length in pixels
    cx, cy : principal point in pixels
    width, height : image size in pixels
    baseline : metres between optical centres, right camera at ``+x_c``
    p_bc : lever arm from the body origin to the left optical centre, body frame
    min_depth, max_depth : depth gate, metres. Below ``min_depth`` the rig sees
        the bonnet; above ``max_depth`` stereo disparity is smaller than the
        matching noise and the triangulated point is worse than useless.
    """

    fx: float = 718.856  # KITTI 00-02 rectified intrinsics, so the numbers
    fy: float = 718.856  # transfer when the real loader comes online
    cx: float = 607.193
    cy: float = 185.216
    width: int = 1241
    height: int = 376
    baseline: float = 0.537
    p_bc: np.ndarray = field(default_factory=lambda: np.array([1.6, 0.0, -1.2]))
    min_depth: float = 3.0
    max_depth: float = 60.0

    # -------------------------------------------------------------- geometry
    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def R_bc(self) -> np.ndarray:
        return R_BC

    def T_bc(self) -> tuple[np.ndarray, np.ndarray]:
        """Camera-to-body rigid transform as ``(R, t)``."""
        return R_BC, np.asarray(self.p_bc, float)

    def nav_to_cam(self, R_nb: np.ndarray, p_nb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pose of the left camera in the nav frame, given the body pose."""
        R_nc = R_nb @ R_BC
        p_nc = p_nb + R_nb @ self.p_bc
        return R_nc, p_nc

    # ------------------------------------------------------------ projection
    def project(self, P_c: np.ndarray) -> np.ndarray:
        """Project camera-frame points ``(N,3)`` to left-image pixels ``(N,2)``."""
        P_c = np.atleast_2d(P_c)
        z = P_c[:, 2]
        return np.stack(
            [self.fx * P_c[:, 0] / z + self.cx, self.fy * P_c[:, 1] / z + self.cy], axis=1
        )

    def project_stereo(self, P_c: np.ndarray) -> np.ndarray:
        """Return ``(N,3)`` of ``[u_left, v, u_right]``.

        The rectified pair shares a row, so the right observation adds exactly
        one number: the horizontal coordinate. Disparity is ``u_l - u_r``.
        """
        P_c = np.atleast_2d(P_c)
        z = P_c[:, 2]
        u_l = self.fx * P_c[:, 0] / z + self.cx
        v = self.fy * P_c[:, 1] / z + self.cy
        u_r = self.fx * (P_c[:, 0] - self.baseline) / z + self.cx
        return np.stack([u_l, v, u_r], axis=1)

    def triangulate(self, uvu: np.ndarray) -> np.ndarray:
        """Invert :meth:`project_stereo`. ``(N,3)`` pixels -> ``(N,3)`` metres."""
        uvu = np.atleast_2d(uvu)
        disp = uvu[:, 0] - uvu[:, 2]
        z = self.fx * self.baseline / disp
        x = (uvu[:, 0] - self.cx) * z / self.fx
        y = (uvu[:, 1] - self.cy) * z / self.fy
        return np.stack([x, y, z], axis=1)

    def visible(self, P_c: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask of points inside the depth gate and both image frusta."""
        P_c = np.atleast_2d(P_c)
        ok = (P_c[:, 2] > self.min_depth) & (P_c[:, 2] < self.max_depth)
        if not ok.any():
            return ok
        uvu = self.project_stereo(P_c)
        in_l = (
            (uvu[:, 0] > margin)
            & (uvu[:, 0] < self.width - margin)
            & (uvu[:, 1] > margin)
            & (uvu[:, 1] < self.height - margin)
        )
        in_r = (uvu[:, 2] > margin) & (uvu[:, 2] < self.width - margin)
        return ok & in_l & in_r

    def depth_sigma(self, depth: float, pixel_sigma: float) -> float:
        """One-sigma depth error implied by a disparity error, metres.

        ``z = f B / d`` so ``dz = z^2 / (f B) * dd``. The quadratic is the whole
        reason stereo VO is range-limited and the reason the depth gate exists.
        """
        return depth**2 / (self.fx * self.baseline) * (np.sqrt(2.0) * pixel_sigma)


KITTI_LIKE = StereoRig()
NARROW_BASELINE = StereoRig(baseline=0.12, max_depth=25.0)  # dashcam-class rig
