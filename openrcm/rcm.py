"""Remote Center of Motion (RCM) inverse kinematics for the CR5 + straight instrument.

An RCM constraint says: the instrument shaft must always pass through one fixed point
in space (in surgery, the incision / trocar port), whatever the tip does.

Geometry (all in the `instrument` link frame, whose +z axis is the shaft axis):
    d = R^T (p_rcm - p_shaft_origin)          RCM point expressed in the tool frame
    RCM error   = (d_x, d_y)                    sideways distance of the RCM from the shaft axis
    insertion s = d_z                           where along the shaft the RCM sits

The IK problem solved at every path sample (nonlinear least squares, jaxls):
    minimize    100 * ||tip(q) - target||^2          tip follows the path
              + 0.02 * ||q - q_prev||^2              stay close to the last pose (smooth motion,
                                                     and picks the free tool-roll DOF)
    subject to  (d_x, d_y) = 0                       RCM: equality constraint
                s_min <= s <= s_max                  RCM stays on the thin shaft
                q_min <= q <= q_max                  joint limits

DOF count: the tip position is 3 constraints and the RCM is 2, for 5 in total. With 6 joints,
the robot keeps 1 redundant DOF, which is the roll of the tool about its own shaft.

Without RCM (for comparison) the 2 RCM constraints are replaced by a soft cost
that holds the shaft along a fixed direction, as a "naive" Cartesian tool would.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import yourdfpy

from openrcm import instrument
from openrcm.kinematics import HOME_CFG
from openrcm.paths import RCM_URDF

SHAFT_LINK = "instrument"
TIP_LINK = "tool_tip"
# Allowed RCM position along the shaft, measured from the flange.
S_MIN = instrument.ADAPTER_LENGTH + instrument.INSERTION_MARGIN
S_MAX = instrument.ADAPTER_LENGTH + instrument.SHAFT_LENGTH - instrument.INSERTION_MARGIN


# Feasibility thresholds used by plan_path().
TIP_TOLERANCE = 1e-3  # m
RCM_TOLERANCE = 5e-4  # m
# A joint change this large between samples 2 mm apart signals an IK branch switch
# (scaled with the actual sample spacing, since normal motion grows with it).
BRANCH_JUMP_DEG = 10.0
BRANCH_JUMP_SPACING = 0.002
LIMIT_WARN_DEG = 10.0
SPEED_WARN_RATIO = 0.8


def rcm_residual(p: onp.ndarray, d: onp.ndarray, c: onp.ndarray) -> tuple[float, float]:
    """Distance of the RCM point c from the shaft axis, and where along the shaft it lies.

    p: a point on the shaft axis, d: shaft direction (unit), c: RCM point, all in the SAME frame.
    Returns (e, s) with e = ||(c - p) x d|| and s = (c - p) . d (signed, along d from p).
    """
    d = onp.asarray(d, dtype=onp.float64)
    d = d / onp.linalg.norm(d)
    v = onp.asarray(c, dtype=onp.float64) - onp.asarray(p, dtype=onp.float64)
    return float(onp.linalg.norm(onp.cross(v, d))), float(v @ d)


def _tool_frames(robot: pk.Robot, cfg: jax.Array, shaft_idx: jax.Array, tip_idx: jax.Array):
    poses = robot.forward_kinematics(cfg)
    return jaxlie.SE3(poses[shaft_idx]), jaxlie.SE3(poses[tip_idx])


@jaxls.Cost.factory
def tip_position_cost(vals, robot, var, shaft_idx, tip_idx, target, weight):
    _, tip = _tool_frames(robot, vals[var], shaft_idx, tip_idx)
    return (tip.translation() - target) * weight


@partial(jaxls.Cost.factory, kind="constraint_eq_zero")
def rcm_constraint(vals, robot, var, shaft_idx, tip_idx, rcm_point):
    shaft, _ = _tool_frames(robot, vals[var], shaft_idx, tip_idx)
    d = shaft.inverse() @ rcm_point
    return d[:2]


@partial(jaxls.Cost.factory, kind="constraint_leq_zero")
def insertion_constraint(vals, robot, var, shaft_idx, tip_idx, rcm_point):
    shaft, _ = _tool_frames(robot, vals[var], shaft_idx, tip_idx)
    s = (shaft.inverse() @ rcm_point)[2]
    return jnp.array([S_MIN - s, s - S_MAX])


@jaxls.Cost.factory
def shaft_direction_cost(vals, robot, var, shaft_idx, tip_idx, direction, weight):
    shaft, _ = _tool_frames(robot, vals[var], shaft_idx, tip_idx)
    return (shaft.rotation() @ jnp.array([0.0, 0.0, 1.0]) - direction) * weight


@jdc.jit
def _solve(
    robot: pk.Robot,
    shaft_idx: jax.Array,
    tip_idx: jax.Array,
    target: jax.Array,
    rcm_point: jax.Array,
    direction: jax.Array,
    prev_cfg: jax.Array,
    use_rcm: jdc.Static[bool],
) -> jax.Array:
    var = robot.joint_var_cls(0)
    costs = [
        tip_position_cost(robot, var, shaft_idx, tip_idx, target, 100.0),
        pk.costs.limit_constraint(robot, var),
        pk.costs.rest_cost(var, prev_cfg, weight=0.02),
    ]
    if use_rcm:
        costs += [
            rcm_constraint(robot, var, shaft_idx, tip_idx, rcm_point),
            insertion_constraint(robot, var, shaft_idx, tip_idx, rcm_point),
        ]
    else:
        costs.append(shaft_direction_cost(robot, var, shaft_idx, tip_idx, direction, 10.0))
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([var.with_value(prev_cfg)]),
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[var]


@dataclass
class ToolState:
    """Where the instrument is, and how well the constraints are met."""

    tip: onp.ndarray  # tip position (world)
    shaft_origin: onp.ndarray  # instrument frame origin = flange
    axis: onp.ndarray  # shaft direction (unit, flange -> tip)
    rcm_error: float  # distance of the RCM point from the shaft axis (m)
    insertion: float  # RCM position along the shaft, from the flange (m)
    closest_on_shaft: onp.ndarray  # point of the shaft axis nearest the RCM point

    @property
    def insertion_ok(self) -> bool:
        return S_MIN - 1e-4 <= self.insertion <= S_MAX + 1e-4

    @property
    def depth_past_rcm(self) -> float:
        """How far the tip is beyond the RCM point (the 'inserted' length)."""
        return instrument.TOOL_LENGTH - self.insertion


class RCMKinematics:
    def __init__(self) -> None:
        urdf = yourdfpy.URDF.load(RCM_URDF, load_meshes=False, build_scene_graph=True)
        self.urdf = urdf
        self.robot = pk.Robot.from_urdf(urdf, default_joint_cfg=HOME_CFG)
        names = self.robot.links.names
        self.shaft_idx = jnp.array(names.index(SHAFT_LINK))
        self.tip_idx = jnp.array(names.index(TIP_LINK))
        self.joint_names = list(urdf.actuated_joint_names)
        self.lower = onp.asarray(self.robot.joints.lower_limits)
        self.upper = onp.asarray(self.robot.joints.upper_limits)
        self.velocity_limit = onp.asarray(self.robot.joints.velocity_limits)

    def solve(
        self,
        target: onp.ndarray,
        prev_cfg: onp.ndarray,
        rcm_point: onp.ndarray | None,
        direction: onp.ndarray | None = None,
    ) -> onp.ndarray:
        """IK for the tip at `target`. With rcm_point=None the shaft is held along `direction`."""
        use_rcm = rcm_point is not None
        cfg = _solve(
            self.robot,
            self.shaft_idx,
            self.tip_idx,
            jnp.asarray(target, dtype=jnp.float32),
            jnp.asarray(rcm_point if use_rcm else onp.zeros(3), dtype=jnp.float32),
            jnp.asarray(direction if direction is not None else [0.0, 0.0, -1.0], dtype=jnp.float32),
            jnp.asarray(prev_cfg, dtype=jnp.float32),
            use_rcm,
        )
        return onp.asarray(cfg, dtype=onp.float64)

    def tool_state(self, cfg: onp.ndarray, rcm_point: onp.ndarray) -> ToolState:
        poses = onp.asarray(self.robot.forward_kinematics(jnp.asarray(cfg)), dtype=onp.float64)
        w, x, y, z = poses[int(self.shaft_idx)][:4]  # pose = (wxyz, xyz)
        origin = poses[int(self.shaft_idx)][4:]
        # Third column of the rotation matrix = the shaft (+z) direction.
        axis = onp.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        axis /= onp.linalg.norm(axis)
        error, s = rcm_residual(origin, axis, rcm_point)
        return ToolState(
            tip=poses[int(self.tip_idx)][4:],
            shaft_origin=origin,
            axis=axis,
            rcm_error=error,
            insertion=s,
            closest_on_shaft=origin + s * axis,
        )


@dataclass
class PathReport:
    """Per-sample results of planning a whole path (see plan_path)."""

    q: onp.ndarray  # [N, 6] joint trajectory (rad)
    tip_error: onp.ndarray  # [N] m
    rcm_error: onp.ndarray  # [N] m (distance of the port from the shaft axis)
    insertion_ok: onp.ndarray  # [N] bool: port on the allowed part of the shaft
    speed_ratio: onp.ndarray  # [N] peak joint speed / limit at the requested tip speed
    limit_margin_deg: onp.ndarray  # [N] closest joint-limit distance
    branch_jump: onp.ndarray  # [N] bool
    bad: onp.ndarray  # [N] bool: infeasible sample
    warn: onp.ndarray  # [N] bool: feasible but close to a limit
    dt: float  # s between samples
    plan_seconds: float  # total, including evaluation of every sample
    ik_seconds: float  # time spent in the IK solver only
    rcm_enforced: bool

    @property
    def feasible(self) -> bool:
        return not bool(self.bad.any())

    @property
    def worst_speed_joint(self) -> int:
        dq = onp.vstack([onp.zeros((1, 6)), onp.abs(onp.diff(self.q, axis=0))])
        return int(onp.argmax(dq.max(axis=0)))


def plan_path(
    kin: RCMKinematics,
    points: onp.ndarray,
    normals: onp.ndarray,
    q_start: onp.ndarray,
    rcm_point: onp.ndarray,
    enforce_rcm: bool = True,
    tip_speed: float = 0.03,
    spacing: float = 0.002,
    settle_iterations: int = 80,
) -> PathReport:
    """Solve the IK for every path sample in order (warm-started) and judge the trajectory.

    points/normals: [N, 3] tip targets and surface normals (world), `spacing` m apart.
    rcm_point: the port (world). Its error is always measured; it is only enforced if enforce_rcm.
    The robot first converges onto points[0]; that approach is not part of the report.
    """
    rcm = onp.asarray(rcm_point) if enforce_rcm else None
    q = onp.asarray(q_start, dtype=onp.float64).copy()
    for _ in range(settle_iterations):
        q = kin.solve(points[0], q, rcm, -normals[0])
    t0 = time.perf_counter()
    ik_seconds = 0.0
    Q, tip_err, rcm_err, ins_ok = [], [], [], []
    for p, n in zip(points, normals):
        t_ik = time.perf_counter()
        q = kin.solve(p, q, rcm, -n)
        ik_seconds += time.perf_counter() - t_ik
        state = kin.tool_state(q, rcm_point)
        Q.append(q)
        tip_err.append(onp.linalg.norm(state.tip - p))
        rcm_err.append(state.rcm_error)
        ins_ok.append(state.insertion_ok)
    plan_seconds = time.perf_counter() - t0
    Q, tip_err, rcm_err, ins_ok = map(onp.asarray, (Q, tip_err, rcm_err, ins_ok))

    dt = spacing / tip_speed
    dq = onp.vstack([onp.zeros((1, 6)), onp.abs(onp.diff(Q, axis=0))])
    speed_ratio = (dq / dt / kin.velocity_limit).max(axis=1)
    margin = onp.rad2deg(onp.minimum(Q - kin.lower, kin.upper - Q).min(axis=1))
    jump = onp.rad2deg(dq.max(axis=1)) > BRANCH_JUMP_DEG * max(1.0, spacing / BRANCH_JUMP_SPACING)
    bad = (tip_err > TIP_TOLERANCE) | (speed_ratio > 1.0) | jump
    if enforce_rcm:
        bad |= (rcm_err > RCM_TOLERANCE) | ~ins_ok
    warn = ~bad & ((margin < LIMIT_WARN_DEG) | (speed_ratio > SPEED_WARN_RATIO))
    return PathReport(
        Q,
        tip_err,
        rcm_err,
        ins_ok,
        speed_ratio,
        margin,
        jump,
        bad,
        warn,
        dt,
        plan_seconds,
        ik_seconds,
        enforce_rcm,
    )
