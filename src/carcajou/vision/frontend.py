"""Stereo visual odometry front end.

Pipeline per VO epoch, matching what a real stereo front end does after
rectification:

1. **Track.** Keep landmarks visible in both frames of both epochs. Standing in
   for detect / describe / match / ratio-test / left-right-consistency, whose
   output is exactly this: a set of correspondences, most right, some wrong.
2. **Mask.** Drop correspondences the segmenter labels dynamic. Applied
   *before* pose estimation, which is the whole point; masking after RANSAC has
   already converged on the lead vehicle is too late.
3. **Triangulate.** Disparity to metric 3D in each epoch's camera frame. Stereo
   is what makes the translation metric, so unlike monocular VO there is no
   scale to lose and no scale drift to model.
4. **Align.** RANSAC over 3-point Kabbsch fits between the two point sets. 3D-3D
   alignment rather than PnP: both epochs already have depth, the closed form is
   exact, and the covariance falls out of the same normal equations.
5. **Convert.** Relative camera pose to a body-frame velocity with a covariance
   the filter can trust, which is the only artefact the ESKF ever sees.

Why velocity and not a relative-pose factor
-------------------------------------------
A relative-pose measurement links two epochs, and a filter that consumes one
correctly needs stochastic cloning: augment the state with a copy of the pose at
the earlier epoch so the correlation between the two is carried. That is the
right answer and it is Phase 3 work.

What is implemented here is the pragmatic standard: convert the displacement to
an average body-frame velocity over the interval and feed it as an
instantaneous velocity measurement. This throws away the cross-epoch
correlation and treats successive VO updates as independent when they are not,
which makes the filter mildly overconfident. Two things keep it honest: the
measurement noise is inflated by the average-versus-instantaneous discrepancy
term below, and the resulting optimism is reported in the covariance
consistency plot rather than hidden. It is the same class of approximation the
README already owns up to for GNSS whiteness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..datasets.synthetic import Trajectory
from ..frames import exp_so3, skew
from .camera import StereoRig
from .segmentation import MASK_REALISTIC, SimulatedMask
from .world import LandmarkWorld


@dataclass
class VoConfig:
    """Front-end and noise parameters.

    Attributes
    ----------
    rate_hz : VO update rate. 10 Hz is the automotive norm and matches the
        KITTI packet rate, so the synthetic and real paths stay comparable.
    pixel_sigma : one-sigma feature localisation error. 0.5 px is a good
        sub-pixel corner refiner on a well-textured scene.
    max_features : cap on correspondences per epoch, as every real front end has.
    outlier_rate : fraction of surviving correspondences that are gross
        mismatches, standing in for descriptor aliasing on repetitive structure.
    min_inliers : below this the VO measurement is discarded rather than fed to
        the filter with an optimistic covariance. Refusing to answer is a
        feature; a degenerate VO update during a tunnel transition is worse
        than no update at all.
    ransac_n_sigma : inlier gate in multiples of each correspondence's own
        predicted 3D sigma. A fixed metric gate cannot span a 5 m kerbstone and
        a 50 m facade in the same frame.
    accel_sigma : expected magnitude of acceleration over one VO interval, used
        to inflate R for the average-versus-instantaneous velocity mismatch.
    sigma_floor : hard floor on the per-axis velocity sigma. Stops a degenerate
        geometry from handing the filter an absurdly confident measurement.
    """

    rate_hz: float = 10.0
    pixel_sigma: float = 0.5
    max_features: int = 300
    outlier_rate: float = 0.02
    min_inliers: int = 25
    ransac_iters: int = 120
    ransac_n_sigma: float = 3.0
    ransac_floor: float = 0.05  # metres, so near points still get a real gate
    accel_sigma: float = 2.0
    sigma_floor: float = 0.02
    gate_speed: float = 0.3  # below this the scene is static; ZUPT is better


@dataclass
class VoMeasurement:
    """One VO epoch's output, in the form the ESKF consumes."""

    t: float
    dt: float
    index: int  # trajectory index of the later epoch
    index_prev: int  # trajectory index of the earlier epoch
    v_b: np.ndarray  # body-frame velocity at the later epoch, m/s
    R_v: np.ndarray  # 3x3 measurement covariance
    dR_b: np.ndarray  # relative body rotation, maps curr-body to prev-body
    R_rot: np.ndarray  # 3x3 covariance of the rotation measurement, body frame
    n_tracked: int = 0
    n_after_mask: int = 0
    n_inliers: int = 0
    n_dynamic_kept: int = 0  # dynamic points the segmenter let through
    diagnostics: dict = field(default_factory=dict)


# ------------------------------------------------------------------- geometry
def kabsch(
    A: np.ndarray, B: np.ndarray, w: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least-squares rigid transform with ``B ~ R @ A + t``. No scale.

    Weights matter more here than they do in a textbook. Stereo depth error is
    ``z^2 / (f * baseline)`` per pixel of disparity error, so a point at 50 m is
    two orders of magnitude worse conditioned than one at 5 m. Fitting them with
    equal weight lets the far field, which carries almost no usable range
    information, dominate a solution the near field could have pinned down.
    """
    if w is None:
        w = np.ones(len(A))
    w = w / w.sum()
    ca = w @ A
    cb = w @ B
    H = ((A - ca) * w[:, None]).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cb - R @ ca


def ransac_rigid(
    A: np.ndarray,
    B: np.ndarray,
    sigma: np.ndarray,
    iters: int,
    rng: np.random.Generator,
    n_sigma: float = 3.0,
    floor: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC 3-point rigid fit with a per-point, depth-aware inlier gate.

    ``sigma`` is the combined one-sigma 3D position error of each correspondence
    across the two epochs. A fixed metric threshold cannot work: it either
    rejects every legitimate far-field point or accepts gross mismatches in the
    near field.

    Note what RANSAC can and cannot do here. It rejects *minority* outliers. If
    a lead vehicle owns more of the tracked set than the static world does, the
    dynamic points are the consensus and RANSAC will confidently return the
    wrong pose. That is not a defect in RANSAC; it is the reason segmentation
    has to run first, and it is what the mask ablation measures.
    """
    n = len(A)
    best_inl = np.zeros(n, bool)
    if n < 3:
        return np.eye(3), np.zeros(3), best_inl

    thr = np.maximum(n_sigma * sigma, floor)
    w = 1.0 / np.maximum(sigma, 1e-6) ** 2

    # MSAC rather than vanilla RANSAC: score by the truncated squared
    # Mahalanobis residual instead of raw inlier count. This matters here
    # specifically. A lead vehicle is the nearest object in the frame, so its
    # points are the best-triangulated ones in the frame, and any scoring rule
    # that rewards precision would hand a tight 20-point cluster on a car the
    # win over 200 correctly-tracked facade points. Truncation charges a
    # hypothesis for everything it fails to explain, which is what stops that.
    best_cost = np.inf
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            R, t = kabsch(A[idx], B[idx])
        except np.linalg.LinAlgError:
            continue
        resid = np.linalg.norm(B - (A @ R.T + t), axis=1)
        inl = resid < thr
        cost = float(np.minimum((resid / np.maximum(thr, 1e-9)) ** 2, 1.0).sum())
        if cost < best_cost:
            best_cost, best_inl = cost, inl
            if inl.sum() > 0.95 * n:
                break

    if best_inl.sum() < 3:
        return np.eye(3), np.zeros(3), best_inl

    R, t = kabsch(A[best_inl], B[best_inl], w[best_inl])
    for _ in range(2):  # a couple of reweighted refits; this converges fast
        resid = np.linalg.norm(B - (A @ R.T + t), axis=1)
        inl = resid < thr
        if inl.sum() < 3:
            break
        R, t = kabsch(A[inl], B[inl], w[inl])
        best_inl = inl
    return R, t, best_inl


def rigid_covariance(
    A: np.ndarray, B: np.ndarray, R: np.ndarray, t: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """6x6 covariance of ``[dtheta, dt]`` from the alignment normal equations.

    Residual ``r_i = B_i - (R A_i + t)``. Perturbing ``R <- Exp(dtheta) R`` and
    ``t <- t + dt`` gives ``J_i = [skew(R A_i), -I]``, so the information matrix
    is ``sum w_i J_i^T J_i`` with ``w_i = 1 / sigma_i^2``.

    The scale is then rescaled by the *empirical* normalised chi-square rather
    than trusting ``sigma`` outright. If the front end's noise model is
    optimistic, or the frame is full of half-matched foliage, the residuals say
    so and the reported covariance grows. A VO front end that reports a
    covariance it cannot back up is worse than one that reports none, because
    the filter has no way to find out.
    """
    n = len(A)
    resid = B - (A @ R.T + t)
    w = 1.0 / np.maximum(sigma, 1e-6) ** 2
    dof = max(3 * n - 6, 1)
    chi2_n = float((w[:, None] * resid**2).sum() / dof)
    scale = max(chi2_n, 1.0)  # never talk the covariance down below its model

    RA = A @ R.T
    # Vectorised accumulation of sum w_i J_i^T J_i with J_i = [skew(RA_i), -I].
    S = np.zeros((n, 3, 3))
    S[:, 0, 1], S[:, 0, 2] = -RA[:, 2], RA[:, 1]
    S[:, 1, 0], S[:, 1, 2] = RA[:, 2], -RA[:, 0]
    S[:, 2, 0], S[:, 2, 1] = -RA[:, 1], RA[:, 0]
    info = np.zeros((6, 6))
    info[0:3, 0:3] = np.einsum("n,nji,njk->ik", w, S, S)
    info[0:3, 3:6] = -np.einsum("n,nji->ij", w, S).T
    info[3:6, 0:3] = info[0:3, 3:6].T
    info[3:6, 3:6] = w.sum() * np.eye(3)
    try:
        cov = scale * np.linalg.inv(info + np.eye(6) * 1e-9)
    except np.linalg.LinAlgError:
        cov = np.eye(6) * 1e3
    return 0.5 * (cov + cov.T)


def refine_reprojection(
    rig: StereoRig,
    P_prev: np.ndarray,
    uvu_curr: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    pixel_sigma: float,
    iters: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-Newton refit minimising stereo reprojection error. Returns ``(R, t, cov6)``.

    The 3D-3D alignment above is a good initialiser and a poor estimator. It
    squares the triangulation error into the residual, so a frame's accuracy is
    set by its worst-conditioned depths. Minimising *pixel* error instead keeps
    each measurement in the space it was actually made in: lateral motion and
    rotation are pinned by flow, which is strong, and only the forward component
    leans on disparity, which is weak. That anisotropy is real, it is why stereo
    VO is far better sideways than forwards, and it lands correctly in the
    covariance instead of being averaged away.

    Residual per correspondence is ``[u_l, v, u_r]``, three numbers, so the
    right camera contributes as an independent observation rather than being
    consumed by triangulation and then discarded.

    The loss is Huber, not least squares, and that is load-bearing. The 3D
    RANSAC gate scales with triangulation sigma, which is quadratic in depth,
    so at typical lead-vehicle range the gate is the better part of a metre
    wide and a speed-matched car's points can slip through it. In pixel space
    the same points are tens of pixels wrong under the static-world pose that
    RANSAC initialised, so a robust loss at a few pixels removes their
    influence almost entirely. This is the per-point second line of defence;
    the segmentation mask is the first, and the ablation measures what happens
    when the first is absent.
    """
    n = len(P_prev)
    cov = np.eye(6) * 1e3
    if n < 3:
        return R, t, cov

    fx, fy, B = rig.fx, rig.fy, rig.baseline
    huber_px = 3.0 * max(pixel_sigma, 1e-3)
    w_pt = np.ones(n)
    info = np.eye(6)
    for _ in range(iters):
        X = P_prev @ R.T + t
        z = np.clip(X[:, 2], 1e-3, None)
        pred = np.stack(
            [
                fx * X[:, 0] / z + rig.cx,
                fy * X[:, 1] / z + rig.cy,
                fx * (X[:, 0] - B) / z + rig.cx,
            ],
            axis=1,
        )
        r = uvu_curr - pred

        # Huber IRLS weights on the per-correspondence residual norm.
        rn = np.linalg.norm(r, axis=1)
        w_pt = np.where(rn <= huber_px, 1.0, huber_px / np.maximum(rn, 1e-9))

        RP = P_prev @ R.T
        # Vectorised J_i = Jp_i @ [-skew(RP_i), I], accumulated as
        # info = sum w_i J_i^T J_i and rhs = sum w_i J_i^T r_i.
        Jp = np.zeros((n, 3, 3))
        Jp[:, 0, 0] = fx / z
        Jp[:, 0, 2] = -fx * X[:, 0] / z**2
        Jp[:, 1, 1] = fy / z
        Jp[:, 1, 2] = -fy * X[:, 1] / z**2
        Jp[:, 2, 0] = fx / z
        Jp[:, 2, 2] = -fx * (X[:, 0] - B) / z**2
        S = np.zeros((n, 3, 3))
        S[:, 0, 1], S[:, 0, 2] = -RP[:, 2], RP[:, 1]
        S[:, 1, 0], S[:, 1, 2] = RP[:, 2], -RP[:, 0]
        S[:, 2, 0], S[:, 2, 1] = -RP[:, 1], RP[:, 0]
        J = np.concatenate([-np.einsum("nab,nbc->nac", Jp, S), Jp], axis=2)  # (n,3,6)
        info = np.einsum("n,nab,nac->bc", w_pt, J, J)
        rhs = np.einsum("n,nab,na->b", w_pt, J, r)
        try:
            d = np.linalg.solve(info + np.eye(6) * 1e-9, rhs)
        except np.linalg.LinAlgError:
            break
        R = exp_so3(d[0:3]) @ R
        t = t + d[3:6]
        if np.linalg.norm(d) < 1e-10:
            break

    X = P_prev @ R.T + t
    z = np.clip(X[:, 2], 1e-3, None)
    pred = np.stack(
        [fx * X[:, 0] / z + rig.cx, fy * X[:, 1] / z + rig.cy, fx * (X[:, 0] - B) / z + rig.cx],
        axis=1,
    )
    r = uvu_curr - pred
    rn = np.linalg.norm(r, axis=1)
    w_pt = np.where(rn <= huber_px, 1.0, huber_px / np.maximum(rn, 1e-9))
    dof = max(3 * n - 6, 1)
    chi2_n = float((w_pt[:, None] * r**2).sum() / (max(pixel_sigma, 1e-6) ** 2 * dof))
    scale = max(pixel_sigma**2 * max(chi2_n, 1.0), 1e-12)
    try:
        cov = scale * np.linalg.inv(info + np.eye(6) * 1e-9)
    except np.linalg.LinAlgError:
        cov = np.eye(6) * 1e3
    return R, t, 0.5 * (cov + cov.T)


# ------------------------------------------------------------------ front end
class StereoVo:
    """Synthetic stereo VO over a :class:`LandmarkWorld`."""

    def __init__(
        self,
        rig: StereoRig,
        world: LandmarkWorld,
        traj: Trajectory,
        cfg: VoConfig | None = None,
        mask: SimulatedMask | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.rig = rig
        self.world = world
        self.traj = traj
        self.cfg = cfg or VoConfig()
        self.mask = MASK_REALISTIC if mask is None else mask
        self.rng = rng or np.random.default_rng(0)
        self._tree = cKDTree(world.p_static)
        self.step_indices = self._plan()

    def _plan(self) -> np.ndarray:
        stride = max(1, int(round((1.0 / self.cfg.rate_hz) / self.traj.dt)))
        return np.arange(0, len(self.traj.t), stride)

    # ------------------------------------------------------------- internals
    def _visible_static(self, k: int) -> np.ndarray:
        """Indices of static landmarks visible from trajectory index ``k``."""
        R_nc, p_nc = self.rig.nav_to_cam(self.traj.R[k], self.traj.p[k])
        near = np.asarray(
            self._tree.query_ball_point(p_nc, self.rig.max_depth + 2.0), dtype=int
        )
        if near.size == 0:
            return near
        P_c = (self.world.p_static[near] - p_nc) @ R_nc
        return near[self.rig.visible(P_c)]

    def _to_cam(self, p_nav: np.ndarray, k: int) -> np.ndarray:
        R_nc, p_nc = self.rig.nav_to_cam(self.traj.R[k], self.traj.p[k])
        return (np.atleast_2d(p_nav) - p_nc) @ R_nc

    def _observe(self, P_c: np.ndarray, noisy: bool) -> tuple[np.ndarray, np.ndarray]:
        """Project, add pixel noise, triangulate back.

        Returns both the pixel observations and the triangulated points. The
        pixels are the actual measurement; the triangulation is a derived
        quantity used only to initialise. Keeping both is what lets the refit
        work in pixel space.
        """
        uvu = self.rig.project_stereo(P_c)
        if noisy and self.cfg.pixel_sigma > 0:
            uvu = uvu + self.rng.normal(0.0, self.cfg.pixel_sigma, uvu.shape)
        return uvu, self.rig.triangulate(uvu)

    # ----------------------------------------------------------------- public
    def measure(self, k_prev: int, k_curr: int, noisy: bool = True) -> VoMeasurement | None:
        """Produce one VO measurement between two trajectory indices."""
        cfg = self.rig, self.cfg
        rig, c = cfg
        dt = float(self.traj.t[k_curr] - self.traj.t[k_prev])
        if dt <= 0:
            return None

        # --- track: static landmarks visible from both viewpoints
        s_prev = self._visible_static(k_prev)
        s_curr = self._visible_static(k_curr)
        static_ids = np.intersect1d(s_prev, s_curr, assume_unique=False)
        P_static_prev = self._to_cam(self.world.p_static[static_ids], k_prev)
        P_static_curr = self._to_cam(self.world.p_static[static_ids], k_curr)

        # --- track: dynamic landmarks, same physical points, moved world pose
        d_prev_nav = self.world.actor_points(k_prev)
        d_curr_nav = self.world.actor_points(k_curr)
        if len(d_prev_nav) and len(d_curr_nav) == len(d_prev_nav):
            Dp = self._to_cam(d_prev_nav, k_prev)
            Dc = self._to_cam(d_curr_nav, k_curr)
            vis = rig.visible(Dp) & rig.visible(Dc)
            Dp, Dc = Dp[vis], Dc[vis]
        else:
            Dp = Dc = np.zeros((0, 3))

        A_true = np.vstack([P_static_prev, Dp])
        B_true = np.vstack([P_static_curr, Dc])
        is_dyn = np.concatenate(
            [np.zeros(len(P_static_prev), bool), np.ones(len(Dp), bool)]
        )
        n_tracked = len(A_true)
        if n_tracked < c.min_inliers:
            return None

        # --- mask, before pose estimation
        keep = self.mask.keep(is_dyn, self.rng) if noisy else ~is_dyn
        A_true, B_true, is_dyn = A_true[keep], B_true[keep], is_dyn[keep]
        n_after_mask = len(A_true)
        if n_after_mask < c.min_inliers:
            return None

        # --- detector cap, applied uniformly so masking does not change density
        if n_after_mask > c.max_features:
            sel = self.rng.choice(n_after_mask, c.max_features, replace=False)
            A_true, B_true, is_dyn = A_true[sel], B_true[sel], is_dyn[sel]

        # --- observe
        uvu_a, A = self._observe(A_true, noisy)
        uvu_b, B = self._observe(B_true, noisy)

        # --- gross mismatches: a wrong correspondence is wrong in the image,
        # so corrupt the pixels and let triangulation carry it through.
        if noisy and c.outlier_rate > 0:
            bad = self.rng.random(len(A)) < c.outlier_rate
            if bad.any():
                uvu_b[bad] += self.rng.normal(0.0, 25.0, (int(bad.sum()), 3))
                B = self.rig.triangulate(uvu_b)

        # --- per-correspondence 3D uncertainty, dominated by the depth term
        sig_a = np.array([rig.depth_sigma(z, c.pixel_sigma) for z in A[:, 2]])
        sig_b = np.array([rig.depth_sigma(z, c.pixel_sigma) for z in B[:, 2]])
        sigma = np.sqrt(sig_a**2 + sig_b**2) + 1e-6

        # --- align
        Rc, tc, inl = ransac_rigid(
            A, B, sigma, c.ransac_iters, self.rng, c.ransac_n_sigma, c.ransac_floor
        )
        n_inl = int(inl.sum())
        if n_inl < c.min_inliers:
            return None

        # --- refine in pixel space, which is where the measurement lives
        Rc, tc, cov6 = refine_reprojection(
            rig, A[inl], uvu_b[inl], Rc, tc, max(c.pixel_sigma, 1e-3)
        )

        # --- camera relative pose -> body relative pose, lever arm included
        R_bc, p_bc = rig.T_bc()
        dR_b = R_bc @ Rc @ R_bc.T
        q = Rc @ R_bc.T @ p_bc
        t_b = -R_bc @ q + R_bc @ tc + p_bc

        # Velocity of the body in the body frame at the later epoch.
        v_b = -t_b / dt

        # Covariance transport: t_b depends on both blocks of the camera fit.
        J = np.hstack([R_bc @ skew(q), R_bc])
        cov_tb = J @ cov6 @ J.T
        R_v = cov_tb / dt**2

        # Rotation covariance: dR_b = R_bc Rc R_bc^T, so a camera-frame
        # perturbation phi_c maps to the body as R_bc phi_c.
        R_rot = R_bc @ cov6[0:3, 0:3] @ R_bc.T
        R_rot = 0.5 * (R_rot + R_rot.T)

        # Average-over-interval versus instantaneous-at-epoch mismatch. Under
        # braking this dominates every other term, which is why it is modelled
        # rather than folded into a constant.
        R_v = R_v + np.eye(3) * (c.accel_sigma * dt * 0.5) ** 2
        R_v = R_v + np.eye(3) * c.sigma_floor**2
        R_v = 0.5 * (R_v + R_v.T)

        return VoMeasurement(
            t=float(self.traj.t[k_curr]),
            dt=dt,
            index=k_curr,
            index_prev=k_prev,
            v_b=v_b,
            R_v=R_v,
            dR_b=dR_b,
            R_rot=R_rot,
            n_tracked=n_tracked,
            n_after_mask=n_after_mask,
            n_inliers=n_inl,
            n_dynamic_kept=int(is_dyn.sum()),
            diagnostics={
                "dynamic_inlier_fraction": float(is_dyn[inl].mean()) if n_inl else 0.0,
                "median_depth": float(np.median(A[:, 2])),
            },
        )

    def run(self, noisy: bool = True) -> list[VoMeasurement]:
        """Every VO measurement over the whole trajectory."""
        out: list[VoMeasurement] = []
        ks = self.step_indices
        for a, b in zip(ks[:-1], ks[1:], strict=False):
            m = self.measure(int(a), int(b), noisy=noisy)
            if m is not None:
                out.append(m)
        return out
