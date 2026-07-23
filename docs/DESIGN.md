# Design notes

## 1. Why an error-state filter

A direct EKF on the full navigation state linearises a system whose attitude
lives on SO(3) and whose position can be kilometres from the origin. The
linearisation is only valid near the operating point, and the operating point
keeps moving.

The error-state formulation splits the problem:

- The **nominal state** is propagated by the full nonlinear mechanization. No
  linearisation, no small-angle assumption, no truncation.
- The **error state** is what the Kalman filter actually tracks. It is small by
  construction, kept small by injecting it into the nominal state and resetting
  to zero after every measurement update, so the first-order Jacobians stay
  honest.

This also solves the attitude parametrisation problem. The nominal attitude is
a rotation matrix (or quaternion) with no singularity; the error is a
three-parameter rotation vector that never wraps, because it never gets large.

## 2. Frame and error conventions

Getting these wrong produces a filter that works in simulation and diverges on
hardware, so they are stated once and enforced everywhere.

| Quantity | Convention |
|---|---|
| Navigation frame | Local-level NED, fixed tangent plane at the first fix |
| Body frame | FRD (forward, right, down) |
| Attitude | `R` maps body to nav: `v_n = R v_b` |
| Gravity | `g_n = [0, 0, +g]`, so a level static accelerometer reads `[0, 0, -g]` |
| Attitude error | **Global / left**: `R_true = Exp(dtheta) R_est` |

The global attitude error is the choice that determines every sign in `F`. A
local (right) error, `R_true = R_est Exp(dtheta)`, is equally valid and produces
a different but equivalent `F`. Mixing the two silently is the single most
common way to build a filter that is subtly, stably wrong.

## 3. Error-state dynamics

State ordering: `dx = [dp, dv, dtheta, db_a, db_g]`, 15 elements.

### Velocity error

With `f_meas = f_true + b_a + n_a` and `f_hat = f_meas - b_a_hat`:

```
f_true - f_hat = -db_a - n_a
```

Substituting into `v_dot = R f + g` and expanding `R = (I + [dtheta]x) R_hat`:

```
d(dv)/dt = [dtheta]x R_hat f_hat + R_hat (-db_a - n_a)
         = -[R_hat f_hat]x dtheta - R_hat db_a - R_hat n_a
```

using `[a]x b = -[b]x a`.

### Attitude error

From `R_dot = R [w_ib]x - [w_ie]x R` and the same substitution:

```
[d(dtheta)/dt]x R_hat = R_hat [dw]x + ([dtheta]x [w_ie]x - [w_ie]x [dtheta]x) R_hat
```

The commutator identity `[a]x [b]x - [b]x [a]x = [a x b]x` collapses the second
term, and `R [a]x = [R a]x R` moves the first through `R_hat`:

```
d(dtheta)/dt = -[w_ie]x dtheta - R_hat db_g - R_hat n_g
```

### Assembled

```
        dp      dv       dtheta        db_a    db_g
dp   [   0       I          0            0       0   ]
dv   [   0   -2[w_ie]x  -[R f]x         -R       0   ]
dth  [   0       0     -[w_ie]x          0      -R   ]
dba  [   0       0          0        -I/tau_a    0   ]
dbg  [   0       0          0            0   -I/tau_g]
```

Biases are first-order Gauss-Markov, not random walks. A random-walk bias has
unbounded variance, which is physically wrong for a temperature-stabilised MEMS
part and makes the filter over-trust old bias estimates during long outages.

## 4. What is deliberately not modelled

Being explicit about this is the difference between a simplification and a bug.

- **Transport rate `w_en`.** Zero on a fixed tangent plane by construction.
  Reintroduce it if you move to curvilinear or ECEF mechanization.
- **Coning and sculling.** At 100 Hz with automotive dynamics the residual sits
  well below the MEMS noise floor. It becomes the dominant error term at low
  output rates or with a tactical IMU, which is exactly the regime where the
  KITTI 10 Hz packets live. Flagged in `datasets/kitti.py`.
- **Lever arm** between IMU and GNSS antenna. Real vehicles need it; the
  synthetic generator colocates them. Add it as a fixed `H` offset term.
- **GNSS error correlation.** Modelled as white. Real multipath is strongly
  time-correlated, which is why the urban GNSS grade exists and why the
  chi-square gate matters.

## 5. Why ZUPT and NHC, and why they matter more than they look

During a GNSS outage the position error of an unaided INS grows roughly as
`t^2` through velocity error, and as `t^3` through attitude error feeding into
mis-resolved gravity. Nothing bounds it.

Two constraints cost nothing and change the exponent:

**ZUPT (zero-velocity update).** When the vehicle is stationary the true
velocity is zero, so the observed velocity error is entirely filter error. This
directly observes the accelerometer bias, which is otherwise only weakly
observable. It also resets the integration to a known state, so error growth
restarts from zero rather than compounding.

**NHC (non-holonomic constraint).** A wheeled vehicle does not move sideways or
vertically in its own body frame. Two pseudo-measurements per epoch, no
hardware. Because the constraint is expressed in body axes, its Jacobian
couples into attitude, so it bounds heading drift as well as lateral velocity.

The benchmark exists to quantify this. It is not a small effect: on an
industrial-grade MEMS IMU over a 120 s outage, adding both takes drift from
about 3 % of distance travelled to under 0.5 %, which is the difference between
missing the target spec and clearing it by a factor of two.

## 6. Benchmark methodology

Each ablation resumes from **the same snapshot** of a single GNSS-aided pass:
identical nominal state, identical covariance, identical IMU realisation. The
only variable is which aiding sources survive the outage.

This matters. If each variant re-filtered from scratch, the comparison would be
contaminated by differences in initial alignment convergence, and the
ZUPT-enabled variant would look better partly because it aligned better before
the outage even began. Snapshot resumption isolates the thing being measured.

Windows are spaced across the trajectory so the reported statistics cover
outages beginning during acceleration, cruise, turns and stops. **p95 is
reported alongside the median** because a navigation system is judged on its
worst minutes, not its average one.

## 7. Verification strategy

`tests/test_core.py::test_mechanization_reproduces_truth` is the keystone. The
synthetic generator derives IMU measurements by algebraically inverting the
*discrete* mechanization update, so a noise-free integration must retrace the
reference trajectory to machine precision. Measured closure over 6 km and
560 s: **1.3e-7 m** position, **7.2e-10 m/s** velocity, **2.5e-11 deg**
attitude.

That identity means every metre of drift reported by the benchmark is
attributable to an injected sensor error and nothing else. Without it, a
plausible-looking drift curve could just as easily be an integration bug.

## 8. Roadmap

- **Phase 1** Stereo visual odometry as a velocity/pose update, with a
  segmentation mask so features on moving vehicles are never tracked. Ablate
  the mask on and off.
- **Phase 2** LiDAR odometry (NDT) against a locally built map with dynamic
  returns removed; persist static landmarks and add a map-matching update.
- **Phase 3** ROS2 Humble nodes, C++/Eigen port of the filter hot loop,
  hardware-in-the-loop replay.

## 9. Known limitations, measured

These are stated because a navigation filter that hides them is not usable by
anyone downstream.

**The horizontal covariance is mildly optimistic.** Measured on the aided pass,
roughly 95 to 98 percent of epochs fall inside the reported 3-sigma envelope,
against a nominal 99.7. The vertical channel is consistent. Three causes, in
descending order of contribution:

1. GNSS error is modelled as white. Real multipath is strongly time-correlated,
   so consecutive fixes are not independent and the filter extracts more
   information from them than it should.
2. IMU **scale factor** and **axis misalignment** errors are not in the state
   vector. Only additive bias is estimated, so a real part's scale error leaks
   into the residual as unmodelled process noise.
3. The lever arm between IMU and GNSS antenna is assumed zero.

The fix is to extend the state to 21 or 24 elements (adding scale factor and
misalignment) and to model GNSS error as a first-order Gauss-Markov process
rather than white noise. That is queued behind Phase 1, because vision aiding
buys far more accuracy per unit of work.

**The p95 statistics are computed over 24 outage windows** (6 window positions
times 4 noise seeds) per table cell. That is enough to be indicative and not
enough to be tight. Treat the medians as solid and the p95 values as
directional; widen `--seeds` before quoting them anywhere that matters.

**The KITTI loader has not been validated against a real drive.** Frame
conversions are documented and `validate_against_truth` performs pre-flight
checks on rate, timestamp jitter and gravity convention, but no numbers in this
repository come from KITTI. Do not claim otherwise until that check has been
run.
