"""Tests for RCM geometry, constrained IK and path planning (openrcm.rcm).

Run: pytest tests -q
"""

from __future__ import annotations

import jaxlie
import numpy as onp
import pytest
from scipy.spatial.transform import Rotation

from openrcm import instrument
from openrcm.kinematics import HOME_CFG
from openrcm.rcm import RCM_TOLERANCE, RCMKinematics, plan_path, rcm_residual
from openrcm.rcm_scene import PATTERNS, SURFACES, WorkObject

RNG = onp.random.default_rng(1234)


@pytest.fixture(scope="module")
def kin() -> RCMKinematics:
    return RCMKinematics()


def _port_above_center(obj: WorkObject, height: float = 0.08) -> onp.ndarray:
    return obj.point_world(*obj.path_center_xy(), height)


# ------------------------------------------------------------------ geometry


def test_residual_zero_on_axis():
    p, d = onp.array([0.1, -0.2, 0.3]), onp.array([0.0, 0.6, 0.8])
    e, s = rcm_residual(p, d, p + 0.25 * d)
    assert e == pytest.approx(0.0, abs=1e-12)
    assert s == pytest.approx(0.25)


def test_residual_known_offset():
    p, d = onp.zeros(3), onp.array([0.0, 0.0, 1.0])
    e, s = rcm_residual(p, d, onp.array([0.003, 0.004, 0.2]))  # 3-4-5 triangle: 5 mm off-axis
    assert e == pytest.approx(0.005)
    assert s == pytest.approx(0.2)


def test_residual_ignores_direction_scale_and_sign_of_offset_along_axis():
    p, d = onp.zeros(3), onp.array([1.0, 0.0, 0.0])
    assert rcm_residual(p, 7.0 * d, onp.array([-0.3, 0.01, 0.0]))[0] == pytest.approx(0.01)


def test_residual_random_directions():
    for _ in range(20):
        p = RNG.normal(size=3)
        d = RNG.normal(size=3)
        d /= onp.linalg.norm(d)
        perp = RNG.normal(size=3)
        perp -= (perp @ d) * d
        perp = 0.017 * perp / onp.linalg.norm(perp)  # 17 mm offset
        s_target = float(RNG.uniform(-0.5, 0.5))
        c = p + s_target * d + perp
        e, s = rcm_residual(p, d, c)
        assert e == pytest.approx(0.017, rel=1e-6)
        assert s == pytest.approx(s_target, abs=1e-9)


# ----------------------------------------------------------------------- IK


def test_ik_matches_target_and_rcm_from_home(kin: RCMKinematics):
    obj = WorkObject(shape="Flat pad")
    target = obj.path_world()[0][0]
    port = _port_above_center(obj, 0.08)
    q = kin.solve(target, HOME_CFG, port)
    for _ in range(10):
        q = kin.solve(target, q, port)
    state = kin.tool_state(q, port)
    assert onp.linalg.norm(state.tip - target) < 1e-3
    assert state.rcm_error < RCM_TOLERANCE
    assert state.insertion_ok


def test_ik_respects_joint_limits(kin: RCMKinematics):
    obj = WorkObject(shape="Flat pad")
    target = obj.path_world()[0][0]
    port = _port_above_center(obj, 0.08)
    q = kin.solve(target, HOME_CFG, port)
    assert (q >= kin.lower - 1e-4).all()
    assert (q <= kin.upper + 1e-4).all()


def test_insertion_depth_tracked(kin: RCMKinematics):
    obj = WorkObject()
    target = obj.path_world()[0][0]
    # Ports at two different heights: higher port -> smaller insertion (port sits near tip).
    q_low = kin.solve(target, HOME_CFG, _port_above_center(obj, 0.06))
    q_high = kin.solve(target, HOME_CFG, _port_above_center(obj, 0.12))
    s_low = kin.tool_state(q_low, _port_above_center(obj, 0.06)).insertion
    s_high = kin.tool_state(q_high, _port_above_center(obj, 0.12)).insertion
    # Insertion depth is measured from the flange; a port closer to the surface
    # means the shaft must reach deeper into the port to touch the surface.
    assert s_low > s_high


# ------------------------------------------------------------------ planning


@pytest.mark.parametrize("surface", SURFACES)
def test_circle_feasible_on_all_surfaces(kin: RCMKinematics, surface: str):
    obj = WorkObject(shape=surface)
    obj.pattern = "Circle"
    pts, normals = obj.path_world()
    port = _port_above_center(obj, 0.08)
    rep = plan_path(kin, pts, normals, HOME_CFG, port, enforce_rcm=True, tip_speed=0.03)
    assert rep.feasible, f"{surface} Circle failed: bad samples = {int(rep.bad.sum())}"
    assert rep.rcm_error.max() < RCM_TOLERANCE
    assert rep.tip_error.max() < 1e-3
    assert not rep.branch_jump.any()


@pytest.mark.parametrize("pattern", PATTERNS)
def test_all_patterns_feasible_on_dome(kin: RCMKinematics, pattern: str):
    obj = WorkObject(shape="Dome")
    obj.pattern = pattern
    pts, normals = obj.path_world()
    port = _port_above_center(obj, 0.08)
    rep = plan_path(kin, pts, normals, HOME_CFG, port, enforce_rcm=True, tip_speed=0.03)
    assert rep.feasible, f"Dome {pattern} failed: bad samples = {int(rep.bad.sum())}"


def test_without_rcm_incision_error_is_large(kin: RCMKinematics):
    obj = WorkObject(shape="Dome")
    obj.pattern = "Circle"
    pts, normals = obj.path_world()
    port = _port_above_center(obj, 0.08)
    rep_rcm = plan_path(kin, pts, normals, HOME_CFG, port, enforce_rcm=True)
    rep_off = plan_path(kin, pts, normals, HOME_CFG, port, enforce_rcm=False)
    # RCM on: sub-millimeter. RCM off: tip moves around, so the shaft tears sideways through the port.
    assert rep_rcm.rcm_error.max() < 0.0005
    assert rep_off.rcm_error.max() > 0.03  # > 30 mm incision drag
