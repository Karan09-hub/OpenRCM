# OpenRCM — Project Plan

**Status:** Planning  
**Goal:** Build a simulation-first robotics framework combining hand–eye calibration, rigid-body transformations, and remote-center-of-motion (RCM) constrained instrument motion.  
**Audience:** Robotics engineers, surgical robotics developers, and researchers.

## 1. Project Vision

OpenRCM will investigate how robot–camera calibration and coordinate-frame estimation affect constrained robotic motion. A simulated scene will contain a six-axis robot, an instrument, a camera, a rigid calibration target, and a configurable RCM point.

The simulator knows the ground-truth transforms, but calibration algorithms must estimate them from generated observations. Estimates and downstream motion will be evaluated quantitatively against ground truth.

The project should demonstrate engineering depth appropriate for a robotics engineer with 3+ years of experience: explicit mathematical assumptions, modular design, failure handling, reproducible tests, and measured results.

### Version 1 goals
1. A tested rigid-body transformation library.
2. Synthetic eye-in-hand hand–eye calibration.
3. A virtual camera and calibration-target observation pipeline.
4. RCM geometry and constrained trajectory generation.
5. An end-to-end experiment showing how calibration error affects RCM motion.
6. A public repository with reproducible examples, benchmarks, plots, and limitations.

### Out of scope initially
- Clinical use or safety certification.
- Direct control of a real surgical robot.
- Claims that simulation proves real-world accuracy.
- Photorealistic rendering as a prerequisite.
- VLA/deep-learning integration.
- Real camera integration before synthetic validation.

## 2. Coordinate-Frame Convention

Use one convention everywhere:

**`T_AB` maps coordinates expressed in frame B into frame A:**

`p_A = T_AB @ p_B`

A homogeneous transform is:

```text
T = [ R  t ]
    [ 0  1 ]
```

`R` must be a valid rotation matrix and `t` is translation. Use SI units internally (metres and radians), with explicit conversions at interfaces.

Suggested frames:
- `B`: robot base
- `E`: robot end-effector/flange
- `C`: camera
- `T`: calibration target
- `I`: instrument/tool
- `W`: simulation world

Every public function must document transform direction. Do not infer direction from a variable name alone.

## 3. Proposed Architecture

```text
OpenRCM/
├── README.md
├── PLAN.md
├── pyproject.toml
├── geometry/
│   ├── transforms.py
│   ├── rotations.py
│   └── pose_error.py
├── calibration/
│   ├── camera_model.py
│   ├── target_geometry.py
│   ├── target_pose_estimation.py
│   ├── hand_eye_ax_xb.py
│   ├── hand_eye_ax_zb.py
│   └── calibration_metrics.py
├── robot/
│   ├── robot_model.py
│   ├── forward_kinematics.py
│   └── inverse_kinematics.py
├── rcm/
│   ├── rcm_geometry.py
│   ├── constrained_ik.py
│   └── trajectory.py
├── simulation/
│   ├── scene.py
│   ├── virtual_camera.py
│   └── calibration_target.py
├── evaluation/
│   ├── calibration_benchmark.py
│   ├── rcm_benchmark.py
│   └── error_propagation.py
├── examples/
├── tests/
└── docs/
```

This is a target structure, not a requirement to create empty files immediately. Create modules as milestones need them. The mathematical core should run without a GUI.

### Suggested stack
- Python 3.11+ (choose and test a supported version).
- NumPy for transformations and linear algebra.
- SciPy for optimization/numerical methods.
- pytest for automated tests.
- Matplotlib for plots.
- OpenCV for optional camera projection and target-pose estimation.
- A suitable robot kinematics library (for example, PyRoKi) if it fits the chosen robot model.
- Select a visualization backend only after the mathematical baseline works.

## 4. Workstream A — Rigid-Body Transformations

### Tasks
- [ ] Document the `T_AB` convention.
- [ ] Implement transform construction from rotation and translation.
- [ ] Implement composition and inversion.
- [ ] Implement point and vector transformation.
- [ ] Validate shape, finite values, homogeneous bottom row, rotation orthogonality, and determinant.
- [ ] Implement translation and angular pose-error metrics.
- [ ] Add deterministic random-transform test helpers.

### Tests
- [ ] Identity and pure translation.
- [ ] Rotations around principal axes.
- [ ] Composition of known transforms.
- [ ] Inverse round-trip.
- [ ] Point transformation and inverse transformation.
- [ ] Malformed transforms and invalid rotations.
- [ ] Explicit unit tests for frame direction.

**Exit criteria:** Tests pass and the frame convention is explained with a worked example.

## 5. Workstream B — Synthetic Eye-in-Hand Calibration

### Setup and notation
Start with **eye-in-hand** calibration: the camera is rigidly mounted to the robot end-effector, while the calibration target is fixed in the world.

Use:
- `T_BE`: end-effector pose in the robot base frame.
- `T_CT`: target pose expressed in camera coordinates.
- `T_BT`: fixed target pose in the robot base frame.
- `T_EC`: camera pose expressed in the end-effector frame (maps camera coordinates into the end-effector frame).

For each sample, the static-target relationship is:

`T_BT = T_BE @ T_EC @ T_CT`

The simulator knows `T_EC`; the estimator does not.

### Relative-motion formulation
For conventional eye-in-hand calibration, derive paired relative robot and camera motions. With a consistent convention, the equation can be written:

`A_i @ X = X @ B_i`

Here `X` is the fixed hand–eye transform. Derive the precise definitions of `A_i` and `B_i` from the chosen frame convention; do not copy a formula without checking transform directions.

### Tasks
- [ ] Generate a fixed target pose.
- [ ] Choose a ground-truth camera-to-end-effector transform.
- [ ] Generate diverse reachable end-effector poses.
- [ ] Compute ideal camera-to-target poses from the ground-truth scene.
- [ ] Generate paired relative motions.
- [ ] Implement a baseline `AX = XB` solver or integrate a well-understood method.
- [ ] Estimate the hand–eye transform from multiple motion pairs.
- [ ] Compare the estimate with ground truth using translation and angular errors.
- [ ] Detect or report insufficient/degenerate motion.
- [ ] Save seeds and samples for reproducibility.

### Calibration-quality considerations
- Translation-only motion or poorly varied rotations may not constrain the solution sufficiently.
- Include rotations around multiple axes and a useful spread of orientations.
- More samples do not necessarily improve results if they are redundant or biased.
- Distinguish ideal synthetic pose measurements from noisy camera-derived observations.

**Exit criteria:** Valid, diverse, noise-free samples recover the known transform within a documented tolerance. Degenerate cases produce diagnostics. Reconstructed target poses are consistent across samples.

## 6. Workstream C — Virtual Camera and Target Observations

### Goal
Replace ideal camera poses with image observations from a simple pinhole camera and estimate target pose from those observations.

For a camera-frame point `(X, Y, Z)`:

`u = fx * X / Z + cx`  
`v = fy * Y / Z + cy`

Handle points behind the camera, points near `Z = 0`, and points outside the image explicitly.

### Tasks
- [ ] Define camera intrinsics and image dimensions.
- [ ] Define checkerboard/marker target geometry.
- [ ] Project 3D target points into pixels.
- [ ] Add optional pixel noise and visibility checks.
- [ ] Estimate target pose from 2D–3D correspondences using OpenCV PnP or a documented baseline.
- [ ] Compare estimated target pose with ground truth.
- [ ] Visualize image points and reprojection error.
- [ ] Provide deterministic noise-free and noisy examples.

### Metrics
- Reprojection RMS (pixels).
- Target translation error.
- Target rotation error.
- Fraction of visible target points.
- Calibration success rate across repeated trials.

**Exit criteria:** Noise-free projection matches analytical expectations; pose estimation is validated under suitable viewing conditions; invalid observations fail clearly.

## 7. Workstream D — Generalized `AX = ZB` Calibration

Add this only after `AX = XB` is tested.

The expression `A X = Z B` is not self-defining. Before implementation, specify:
1. The exact meaning and direction of each transform.
2. Which transforms are measured and which are unknown.
3. The physical setup (eye-in-hand, eye-to-hand, or another arrangement).
4. The equation derived from the frame graph.
5. Whether the available motions make all unknowns identifiable.

### Tasks
- [ ] Write the derivation in `docs/hand_eye_formulations.md`.
- [ ] Draw the coordinate-frame graph.
- [ ] Derive the equation from absolute-pose relationships.
- [ ] Implement a separate solver.
- [ ] Generate synthetic data with known `X` and `Z`.
- [ ] Test identifiability and degenerate cases.
- [ ] Compare both estimates against ground truth.
- [ ] Document assumptions and limitations.

**Exit criteria:** Derivation, frame diagram, implementation, and tests agree; supported cases recover both unknowns; unobservable cases are identified.

## 8. Workstream E — RCM Geometry

Let `p` be a point on the instrument shaft, `d` a unit vector along the shaft, and `c` the remote-center point. The perpendicular distance from the RCM point to the infinite shaft axis is:

`e_RCM = || (c - p) × d ||`

This measures whether the RCM lies on the shaft axis. It does not, by itself, prove correct insertion depth, desired orientation, or collision-free motion.

### Tasks
- [ ] Define instrument frame and shaft axis.
- [ ] Express the shaft and RCM point in the same coordinate frame.
- [ ] Implement the geometric residual and useful diagnostics.
- [ ] Test exact alignment and known offsets.
- [ ] Test invariance under consistent coordinate-frame changes.
- [ ] Implement a geometric constrained-motion baseline.
- [ ] Parameterize insertion/retraction and pivoting.
- [ ] Evaluate RCM residual at every trajectory sample.

**Exit criteria:** Exact aligned cases yield near-zero residual; known offsets match analytical values; tests cover insertion, pivoting, combined motion, and frame conversions.

## 9. Workstream F — Constrained IK and Trajectory Generation

Generate robot configurations and trajectories that reach a desired instrument pose while maintaining RCM and respecting joint limits.

Possible methods:
- Geometric construction for suitable robot geometry.
- Optimization-based IK with explicit constraints.
- Penalty-based optimization as a comparison, clearly labelled as a soft constraint.

A penalty term can reduce RCM error but does not guarantee zero error.

### Tasks
- [ ] Select and document robot model and joint conventions.
- [ ] Validate forward kinematics independently.
- [ ] Define target poses and joint limits.
- [ ] Implement constrained IK with convergence/failure diagnostics.
- [ ] Handle joint limits.
- [ ] Generate insertion/retraction trajectories.
- [ ] Generate pivoting trajectories around RCM.
- [ ] Generate combined insertion and pivoting.
- [ ] Validate intermediate points, not just endpoints.
- [ ] Add timing/smoothing after geometric feasibility works.
- [ ] Add collision checks as a later extension.

### Metrics
- Maximum and RMS RCM error.
- Instrument position and orientation error.
- Joint-limit violations.
- Solver convergence rate.
- Solve/planning time.
- Trajectory smoothness.
- Collision rate, if collision checking is implemented.

**Exit criteria:** Accepted trajectories satisfy a declared RCM tolerance and joint limits; unreachable targets and failures are reported honestly.

## 10. Workstream G — Calibration-to-RCM Error Propagation

This is a key differentiator: connect calibration and constrained motion in one measurable experiment.

### Experiment
1. Generate a ground-truth camera-to-end-effector transform.
2. Estimate it from synthetic robot/camera observations.
3. Observe a target or desired instrument pose in camera coordinates.
4. Transform the pose into the robot base frame using the estimate.
5. Plan RCM-constrained motion to the resulting target.
6. Compare against motion using ground-truth calibration.
7. Repeat with controlled calibration perturbations and observation noise.

### Tasks
- [ ] Perturb calibration translation and rotation in controlled ways.
- [ ] Sweep error over documented ranges.
- [ ] Measure target-pose and RCM errors.
- [ ] Separate calibration error from IK and trajectory discretization error.
- [ ] Repeat with fixed random seeds.
- [ ] Plot calibration error against downstream motion error.
- [ ] Report configuration-dependent and nonlinear behavior.

**Exit criteria:** Reproducible experiments and plots with units, sample counts, and test settings.

## 11. Workstream H — Visualization

Start with script-based 3D visualization; add an interactive UI after the workflow is reproducible.

Minimum useful display:
- Robot, instrument, camera frame/frustum, and target.
- Coordinate axes with unambiguous labels.
- Ground-truth and estimated transforms.
- RCM point and instrument shaft axis.
- Target-pose controls.
- Calibration status and residuals.
- RCM trajectory preview and error plot.

### Tasks
- [ ] Visualize frames and target geometry.
- [ ] Show ground truth and estimates distinctly.
- [ ] Display calibration and RCM errors.
- [ ] Add interactive controls only after the script workflow works.
- [ ] Keep visualization optional for tests.

**Exit criteria:** A reviewer can reproduce calibration and motion experiments from documented commands; tests run headlessly.

## 12. Benchmarking

Create repeatable tests, not just one successful demo.

### Calibration variables
- Number of robot poses.
- Rotation and translation diversity.
- Pixel and robot-pose noise.
- Outliers, if supported.
- Camera intrinsic error.
- Solver initialization, where applicable.

### RCM variables
- Target pose and RCM location.
- Initial robot configuration.
- Proximity to joint limits.
- Trajectory type.
- Calibration quality.
- Solver method and parameters.

### Report
- Translation and rotation error.
- Reprojection RMS.
- Maximum and RMS RCM error.
- Solve/planning time.
- Convergence/failure rate.
- Plots with units and test settings.
- Environment details needed for reproduction.

Never invent benchmark numbers for the README or resume. Generate results from included scripts.

## 13. Testing Strategy

### Unit tests
- Transform composition, inversion, and frame direction.
- Rotation validity and pose-error calculations.
- Camera projection.
- Target pose estimation.
- Hand–eye equation consistency.
- RCM residual and frame invariance.
- Trajectory constraint checks.

### Integration tests
- Ground-truth poses → relative motions → hand–eye estimate.
- Target pose → image projection → PnP estimate.
- Estimated hand–eye transform → base-frame target pose.
- Target pose → constrained IK → RCM-checked trajectory.
- Noisy calibration → downstream RCM error benchmark.

Use deterministic seeds and numerical tolerances rather than exact floating-point equality.

## 14. Development Milestones

### Milestone 1 — Transform library
Deliver: transformation utilities, pose errors, unit tests, frame-convention documentation.

### Milestone 2 — Noise-free hand–eye calibration
Deliver: synthetic poses, relative-motion construction, `AX = XB` solver, ground-truth comparison.

### Milestone 3 — Virtual camera
Deliver: pinhole projection, target geometry, noisy image points, PnP estimation, reprojection metrics.

### Milestone 4 — RCM geometry and constrained trajectories
Deliver: shaft-axis residual, robot FK integration, constrained-IK baseline, trajectory validation.

### Milestone 5 — End-to-end error propagation
Deliver: comparison of motion using estimated versus ground-truth calibration, with reproducible plots.

### Milestone 6 — Generalized `AX = ZB`
Deliver: derivation, frame graph, defined setup, synthetic tests, identifiability analysis.

### Milestone 7 — Interactive visualization
Deliver: scene, controls, calibration and RCM plots.

### Milestone 8 — Optional real-camera extension
After the synthetic baseline is reliable, capture real target images, estimate target poses, compare results against references, and document real-world limitations.

## 15. Repository Quality

Before a public release:
- [ ] README explains problem, architecture, assumptions, and limitations.
- [ ] Include coordinate-frame diagram.
- [ ] Document installation, examples, and test commands.
- [ ] Include one calibration example and one RCM trajectory example.
- [ ] Include benchmark scripts and generated plots.
- [ ] Document supported calibration setups.
- [ ] Explain simulated ground truth versus estimated observations.
- [ ] Pin or test dependencies.
- [ ] Add formatting/linting and automated tests.
- [ ] Choose a suitable open-source license.
- [ ] Credit external models, assets, and libraries according to their licenses.
- [ ] Keep employer proprietary code, data, credentials, and unpublished designs out of the repository.

## 16. Version 1 Definition of Done

- [ ] Transform conventions are explicit and tested.
- [ ] Noise-free hand–eye calibration recovers ground truth for valid, diverse motions.
- [ ] Degenerate calibration samples produce diagnostics.
- [ ] Virtual-camera observations and target pose estimates are validated.
- [ ] RCM geometry is independently tested.
- [ ] At least one constrained-trajectory method reports failures honestly.
- [ ] End-to-end experiments measure calibration-to-RCM error propagation.
- [ ] Examples and tests run from documented commands.
- [ ] All published numbers are produced by included scripts.
- [ ] README states this is an engineering simulation, not a clinically validated system.

## 17. Resume Positioning

After implementation and benchmarking, a truthful entry could be shaped around:

**OpenRCM — Hand–Eye Calibration and RCM-Constrained Robotic Motion**  
*Python, NumPy, SciPy, OpenCV, Robot Kinematics, Optimization*

- Developed a simulation framework to estimate robot–camera transformations from synthetic observations and validate calibration against ground truth.
- Implemented RCM geometry and constrained instrument trajectories, evaluating constraint error, solver convergence, and joint-limit feasibility.
- Quantified how calibration uncertainty affects target localization and downstream RCM-constrained motion using reproducible benchmarks.

These are target bullets, not claims of completed work. Update them to reflect implemented methods and measured results.

## 18. Immediate Next Steps

Start with only these:
1. Create the repository and Python environment.
2. Implement and test homogeneous transformations using the `T_AB` convention.
3. Generate a known eye-in-hand synthetic setup.
4. Implement relative-motion generation and an `AX = XB` baseline.
5. Recover the known transform and report translation/rotation error.
6. Add the virtual camera after noise-free calibration tests pass.

**Engineering rule:** Every transformation must have a documented source frame and destination frame. Many calibration bugs are frame-convention errors, not failures of the underlying mathematics.
