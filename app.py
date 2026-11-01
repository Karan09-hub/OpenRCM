"""OpenRCM: a simulator for motion planning under Remote Center of Motion constraints.

A Dobot CR5 collaborative robot holds a straight surgical-style instrument. You pick
the RCM point (the "incision"), place and orient the work object, and choose a path
drawn on its surface. The tip follows the path while the shaft is kept pivoting through
the RCM point, using inverse kinematics with an RCM equality constraint (openrcm/rcm.py).
Turn the constraint off to visualize what an unconstrained / naive controller would do to
the incision site.

Run:
    python app.py          # http://localhost:8080
    python app.py --port 8085
"""

from __future__ import annotations

import threading
import time
from collections import deque
from functools import partial

from openrcm.env_check import require_venv

require_venv()

import jaxlie
import numpy as onp
import tyro
import viser
import viser.uplot
import yourdfpy
from loguru import logger
from viser.extras import ViserUrdf

from openrcm.kinematics import HOME_CFG
from openrcm.paths import RCM_URDF
from openrcm.rcm import S_MAX, S_MIN, RCMKinematics, plan_path
from openrcm.rcm_scene import PATTERNS, SURFACES, WorkObject

CONTROL_DT = 0.02
PATH_SPACING = 0.002  # m between path samples (openrcm.rcm_scene resamples to this)
TIP_SPEED_LIMIT = 0.15  # m/s: how fast the commanded tip may jump to a new target
RCM_SPEED_LIMIT = 0.05  # m/s: how fast the commanded RCM may move to a new port
SHAFT_TURN_LIMIT = 1.0  # rad/s: how fast the shaft may re-orient when the RCM is off
DEFAULT_PORT_HEIGHT_MM = 80.0
JOINT_COLORS = ("#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4")
OK, WARN, BAD = (40, 180, 90), (255, 165, 0), (220, 40, 40)

LESSON = """
**Remote Center of Motion (RCM).** In keyhole surgery the instrument enters the body through a
small incision. Whatever the tip does, the shaft must **pivot about that point** and never
push sideways on it.

**The math.** At every step the IK solves for the joint angles q that put the tip on the path,
**subject to** the shaft axis passing through the RCM point (2 equality constraints) and the
point staying on the shaft (insertion limits).

**DOF count.** Tip position (3) + RCM (2) = **5 task DOF**. The CR5 has **6 joints**, which
leaves **1 redundant DOF**: the tool's roll about its own shaft.

**Try:** drag the red RCM ball, move or rotate the object, change the path, and switch
*Enforce RCM* off to watch the red "incision drag" trail.
"""


def turn_toward(current: onp.ndarray, goal: onp.ndarray, max_angle: float) -> onp.ndarray:
    """Rotate unit vector `current` toward `goal` by at most `max_angle` radians."""
    dot = float(onp.clip(current @ goal, -1.0, 1.0))
    angle = float(onp.arccos(dot))
    if angle <= max_angle or angle < 1e-7:
        return goal.copy()
    sin_a = onp.sin(angle)
    if sin_a < 1e-7:
        return goal.copy()
    v = current * onp.cos(max_angle) + (goal - current * dot) * (onp.sin(max_angle) / sin_a)
    norm = onp.linalg.norm(v)
    return v / norm if norm > 1e-7 else goal.copy()


def step_toward(current: onp.ndarray, goal: onp.ndarray, max_step: float) -> onp.ndarray:
    delta = goal - current
    dist = float(onp.linalg.norm(delta))
    return goal.copy() if dist <= max_step else current + delta * (max_step / dist)


class RCMApp:
    def __init__(self, port: int) -> None:
        self.kin = RCMKinematics()
        self.obj = WorkObject()
        self.lock = threading.Lock()
        self.pending: list = []  # callables queued by GUI threads, run in the main loop

        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+z")
        self.server.initial_camera.position = (0.35, 0.85, 0.7)
        self.server.initial_camera.look_at = (-0.35, 0.0, 0.2)
        self.server.scene.add_grid("/floor", width=2.0, height=2.0, cell_size=0.1)
        vis_urdf = yourdfpy.URDF.load(
            RCM_URDF, filename_handler=partial(yourdfpy.filename_handler_relative, dir=RCM_URDF.parent)
        )
        self.robot_vis = ViserUrdf(self.server, vis_urdf, root_node_name="/robot")

        self.path_pts, self.path_normals = self.obj.path_world()
        self.path_status: onp.ndarray | None = None  # per-point colors from "Check path"
        self.u = 0.0  # position along the path (sample index, fractional)
        self.direction = 1.0
        self.trail_tip: deque = deque(maxlen=600)
        self.trail_rcm: deque = deque(maxlen=600)
        self.history: deque = deque(maxlen=250)  # (t, rcm_err_mm, tip_err_mm)
        self.t = 0.0
        self.picking = False

        self._build_scene()
        self._build_gui()
        self.rcm_desired = self.obj.point_world(0.0, 0.0, DEFAULT_PORT_HEIGHT_MM / 1000)
        self.rcm_handle.position = self.rcm_desired
        self._initial_pose()

    # ---------------------------------------------------------------- setup

    def _initial_pose(self) -> None:
        """Converge to the path start with the RCM enforced (done once, before showing)."""
        logger.info("Compiling IK and moving to the path start...")
        self.q = HOME_CFG.copy()
        for _ in range(60):
            self.q = self.kin.solve(self.path_pts[0], self.q, None, -self.path_normals[0])
        for _ in range(60):
            self.q = self.kin.solve(self.path_pts[0], self.q, self.rcm_desired)
        self.tip_cmd = self.path_pts[0].copy()
        self.rcm_cmd = self.rcm_desired.copy()
        self.dir_cmd = self.kin.tool_state(self.q, self.rcm_desired).axis
        self.robot_vis.update_cfg(self.q)

    def _build_scene(self) -> None:
        scene = self.server.scene
        self.obj_frame = scene.add_frame(
            "/object",
            show_axes=False,
            position=self.obj.pose.translation(),
            wxyz=self.obj.pose.rotation().wxyz,
        )
        self._refresh_object_mesh()
        self.obj_handle = scene.add_transform_controls(
            "/object_handle",
            scale=0.12,
            line_width=2.0,
            opacity=0.7,
            position=self.obj.pose.translation(),
            wxyz=self.obj.pose.rotation().wxyz,
        )
        self.obj_handle.on_update(lambda _: self._queue(self._on_object_moved))
        self.rcm_handle = scene.add_transform_controls(
            "/rcm",
            scale=0.08,
            disable_rotations=True,
            depth_test=False,
        )
        self.rcm_ball = scene.add_icosphere("/rcm/ball", radius=0.01, color=BAD)
        self.rcm_handle.on_update(lambda _: self._queue(self._on_rcm_moved))
        self.target_ball = scene.add_icosphere("/target", radius=0.004, color=(255, 220, 0))

    def _build_gui(self) -> None:
        gui = self.server.gui
        with gui.add_folder("Lesson", expand_by_default=True):
            gui.add_markdown(LESSON)
        with gui.add_folder("RCM"):
            self.rcm_on = gui.add_checkbox("Enforce RCM", True)
            self.rcm_on.on_update(lambda _: self._queue(self._on_rcm_toggled))
            self.port_height = gui.add_slider("Port height (mm)", 20.0, 200.0, 1.0, DEFAULT_PORT_HEIGHT_MM)
            gui.add_button("Place RCM: click the object").on_click(lambda _: self._start_picking())
            gui.add_button("RCM above path center").on_click(lambda _: self._queue(self._rcm_above_center))
            self.rcm_err_text = gui.add_text("RCM error (mm)", "", disabled=True)
            self.depth_text = gui.add_text("Inserted past RCM (mm)", "", disabled=True)
            self.shaft_text = gui.add_text("RCM on shaft range", "", disabled=True)
        with gui.add_folder("Object"):
            self.surface = gui.add_dropdown("Surface", SURFACES, initial_value=self.obj.shape)
            self.surface.on_update(lambda _: self._queue(self._on_surface_changed))
            self.show_handle = gui.add_checkbox("Show move handle", True)
            self.show_handle.on_update(lambda _: setattr(self.obj_handle, "visible", self.show_handle.value))
            gui.add_button("Reset object pose").on_click(lambda _: self._queue(self._reset_object))
        with gui.add_folder("Path"):
            self.pattern = gui.add_dropdown("Pattern", PATTERNS, initial_value=self.obj.pattern)
            self.size = gui.add_slider("Size (mm)", 20.0, 200.0, 1.0, self.obj.size * 1000)
            self.offset_x = gui.add_slider("Center x (mm)", -80.0, 80.0, 1.0, 0.0)
            self.offset_y = gui.add_slider("Center y (mm)", -80.0, 80.0, 1.0, 0.0)
            for handle in (self.pattern, self.size, self.offset_x, self.offset_y):
                handle.on_update(lambda _: self._queue(self._on_path_changed))
        with gui.add_folder("Motion"):
            self.play = gui.add_button("Play", color="green")
            self.play.on_click(lambda _: self._toggle_play())
            self.playing = False
            self.speed = gui.add_slider("Tip speed (mm/s)", 5.0, 150.0, 1.0, 30.0)
            gui.add_button("Restart path").on_click(lambda _: self._queue(self._restart))
            gui.add_button("Clear trails").on_click(lambda _: self._queue(self._clear_trails))
            self.tip_err_text = gui.add_text("Tip error (mm)", "", disabled=True)
            self.jspeed_text = gui.add_text("Max joint speed", "", disabled=True)
            self.progress = gui.add_progress_bar(0.0)
        with gui.add_folder("Motion planning check"):
            gui.add_button("Check whole path", color="blue").on_click(lambda _: self._queue(self._check_path))
            self.check_md = gui.add_markdown(
                "_Plans the whole path offline and colors it: green = OK, "
                "orange = close to a limit, red = infeasible._"
            )
            self.joint_plot = gui.add_uplot(
                data=(onp.zeros(2), *[onp.zeros(2)] * 6),
                series=(
                    viser.uplot.Series(label="path %"),
                    *[viser.uplot.Series(label=f"J{i + 1}", stroke=c, width=1.5) for i, c in enumerate(JOINT_COLORS)],
                ),
                title="Joint angles along the path (deg)",
                aspect=1.6,
                scales={"x": viser.uplot.Scale(time=False, auto=True)},
            )
        with gui.add_folder("Live errors"):
            self.err_plot = gui.add_uplot(
                data=(onp.zeros(2), onp.zeros(2), onp.zeros(2)),
                series=(
                    viser.uplot.Series(label="t (s)"),
                    viser.uplot.Series(label="RCM error mm", stroke="#e6194b", width=1.5),
                    viser.uplot.Series(label="tip error mm", stroke="#4363d8", width=1.5),
                ),
                title="Errors (mm)",
                aspect=1.6,
                scales={"x": viser.uplot.Scale(time=False, auto=True)},
            )
        self.status = gui.add_markdown("")

    # ------------------------------------------------------------ GUI events

    def _queue(self, fn) -> None:
        with self.lock:
            self.pending.append(fn)

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.play.label = "Pause" if self.playing else "Play"
        self.play.color = "orange" if self.playing else "green"

    def _start_picking(self) -> None:
        if self.picking:
            return
        self.picking = True
        self.status.content = "**Click a point on the object surface to place the RCM port above it.**"

        @self.server.scene.on_pointer_event(event_type="click")
        def _(event: viser.ScenePointerEvent) -> None:
            if event.ray_origin is None:
                return
            hit = self.obj.raycast(onp.array(event.ray_origin), onp.array(event.ray_direction))
            if hit is None:
                self.status.content = "_Missed the object, click on its surface._"
                return
            self.server.scene.remove_pointer_callback()
            self.picking = False
            self._queue(lambda: self._set_rcm(self.obj.point_world(*hit, self.port_height.value / 1000)))
            self.status.content = ""

    def _set_rcm(self, p: onp.ndarray) -> None:
        self.rcm_desired = onp.asarray(p, dtype=onp.float64)
        self.rcm_handle.position = self.rcm_desired
        self.path_status = None

    def _rcm_above_center(self) -> None:
        self._set_rcm(self.obj.point_world(*self.obj.path_center_xy(), self.port_height.value / 1000))

    def _on_rcm_moved(self) -> None:
        self.rcm_desired = onp.asarray(self.rcm_handle.position, dtype=onp.float64)
        self.path_status = None

    def _on_rcm_toggled(self) -> None:
        state = self.kin.tool_state(self.q, self.rcm_desired)
        self.rcm_cmd = state.closest_on_shaft
        self.dir_cmd = state.axis
        self.path_status = None

    def _on_object_moved(self) -> None:
        self.obj.pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(onp.asarray(self.obj_handle.wxyz)), onp.asarray(self.obj_handle.position)
        )
        self.obj_frame.position, self.obj_frame.wxyz = self.obj_handle.position, self.obj_handle.wxyz
        self._refresh_path()
        self.u = 0.0
        self.direction = 1.0
        self.tip_cmd = self.path_pts[0].copy()
        self._clear_trails()

    def _reset_object(self) -> None:
        self.obj_handle.position, self.obj_handle.wxyz = (-0.42, 0.0, 0.10), (1.0, 0.0, 0.0, 0.0)
        self._on_object_moved()

    def _on_surface_changed(self) -> None:
        self.obj.shape = self.surface.value
        self._refresh_object_mesh()
        self._refresh_path()
        self.u = 0.0
        self.direction = 1.0
        self.tip_cmd = self.path_pts[0].copy()
        self._clear_trails()

    def _on_path_changed(self) -> None:
        self.obj.pattern = self.pattern.value
        self.obj.size = self.size.value / 1000
        self.obj.path_offset = onp.array([self.offset_x.value, self.offset_y.value]) / 1000
        self._refresh_path()
        self.u = 0.0
        self.direction = 1.0
        self.tip_cmd = self.path_pts[0].copy()
        self._clear_trails()

    def _restart(self) -> None:
        self.u, self.direction = 0.0, 1.0
        self.tip_cmd = self.path_pts[0].copy()
        self._clear_trails()

    def _clear_trails(self) -> None:
        self.trail_tip.clear()
        self.trail_rcm.clear()

    def _refresh_object_mesh(self) -> None:
        mesh = self.obj.mesh_local()
        self.server.scene.add_mesh_simple(
            "/object/mesh",
            mesh.vertices.astype(onp.float32),
            mesh.faces.astype(onp.uint32),
            color=(235, 160, 150),
            flat_shading=False,
        )

    def _refresh_path(self) -> None:
        self.path_pts, self.path_normals = self.obj.path_world()
        self.path_status = None

    # ------------------------------------------------------- planning check

    def _check_path(self) -> None:
        """Offline plan of the whole path (openrcm.rcm.plan_path), shown on the path and in a plot."""
        self.check_md.content = "_Planning..._"
        r = plan_path(
            self.kin,
            self.path_pts,
            self.path_normals,
            self.q,
            self.rcm_desired,
            enforce_rcm=self.rcm_on.value,
            tip_speed=self.speed.value / 1000,
            spacing=PATH_SPACING,
        )
        self.path_status = onp.where(r.bad[:, None], BAD, onp.where(r.warn[:, None], WARN, OK))
        self.joint_plot.data = (onp.linspace(0, 100, len(r.q)), *[onp.rad2deg(r.q[:, j]) for j in range(6)])
        if not r.feasible:
            verdict = f"**Infeasible at {r.bad.mean() * 100:.0f}% of the path**"
        elif not r.rcm_enforced:
            verdict = f"**Tip path feasible, but the port is dragged {r.rcm_error.max() * 1000:.0f} mm**"
        else:
            verdict = "**Feasible**"
        rcm_line = (
            f"- worst RCM error: **{r.rcm_error.max() * 1000:.3f} mm**, insertion in range: "
            f"**{r.insertion_ok.mean() * 100:.0f}%**\n"
            if r.rcm_enforced
            else f"- RCM **off**: the shaft drags up to **{r.rcm_error.max() * 1000:.1f} mm** at the port\n"
        )
        self.check_md.content = (
            f"{verdict}: {len(r.q)} samples, {len(r.q) * r.dt:.1f} s at {self.speed.value:.0f} mm/s "
            f"(planned in {r.plan_seconds:.1f} s)\n\n"
            f"- worst tip error: **{r.tip_error.max() * 1000:.3f} mm**\n"
            + rcm_line
            + f"- peak joint speed: **{r.speed_ratio.max() * 100:.0f}%** of limit (J{r.worst_speed_joint + 1})\n"
            f"- closest approach to a joint limit: **{r.limit_margin_deg.min():.1f} deg**\n"
            f"- IK branch jumps: **{int(r.branch_jump.sum())}**"
        )

    # ------------------------------------------------------------ main loop

    def _advance_path(self) -> None:
        n = len(self.path_pts)
        closed = onp.linalg.norm(self.path_pts[0] - self.path_pts[-1]) < 2 * PATH_SPACING
        self.u += self.direction * self.speed.value / 1000 * CONTROL_DT / PATH_SPACING
        if closed:
            self.u %= n - 1
        elif self.u >= n - 1:
            self.u = float(n - 1)
            self.direction = -1.0
        elif self.u <= 0:
            self.u = 0.0
            self.direction = 1.0

    def _path_point(self) -> tuple[onp.ndarray, onp.ndarray]:
        n = len(self.path_pts)
        u_clamped = float(onp.clip(self.u, 0.0, n - 1.0001))
        i = int(u_clamped)
        f = u_clamped - i
        p = self.path_pts[i] * (1.0 - f) + self.path_pts[i + 1] * f
        return p, self.path_normals[i]

    def tick(self) -> None:
        with self.lock:
            jobs, self.pending = self.pending, []
        for job in jobs:
            job()
        if self.playing:
            self._advance_path()
        target, normal = self._path_point()
        self.tip_cmd = step_toward(self.tip_cmd, target, TIP_SPEED_LIMIT * CONTROL_DT)
        rcm = None
        if self.rcm_on.value:
            self.rcm_cmd = step_toward(self.rcm_cmd, self.rcm_desired, RCM_SPEED_LIMIT * CONTROL_DT)
            rcm = self.rcm_cmd
        else:
            self.dir_cmd = turn_toward(self.dir_cmd, -normal, SHAFT_TURN_LIMIT * CONTROL_DT)
        q_new = self.kin.solve(self.tip_cmd, self.q, rcm, self.dir_cmd)
        delta_q = q_new - self.q
        max_step = self.kin.velocity_limit * CONTROL_DT
        # Branch-jump rejection: if any joint attempts to jump > 25 deg in one tick, reject to prevent wild flailing
        if float(onp.max(onp.abs(delta_q))) < 0.45:
            step = onp.clip(delta_q, -max_step, max_step)
            self.q = self.q + step
            self.joint_speed = onp.abs(step) / CONTROL_DT
        else:
            self.joint_speed = onp.zeros(6)
        self.t += CONTROL_DT
        self.state = self.kin.tool_state(self.q, self.rcm_desired)
        self.target = target
        if self.playing:
            self.trail_tip.append(self.state.tip.copy())
            self.trail_rcm.append(self.state.closest_on_shaft.copy())
        self.history.append((self.t, self.state.rcm_error * 1000, onp.linalg.norm(self.state.tip - target) * 1000))

    def render(self, slow: bool) -> None:
        s = self.state
        with self.server.atomic():
            self.robot_vis.update_cfg(self.q)
            self.target_ball.position = self.target
            self.rcm_ball.color = OK if s.rcm_error < 5e-4 and s.insertion_ok else BAD
            ends = onp.stack([s.shaft_origin, s.shaft_origin + s.axis * (S_MAX + 0.1)])
            self.server.scene.add_line_segments(
                "/shaft_axis", ends[None], colors=(120, 120, 120), thickness=1.0, thickness_units="screen"
            )
            if slow:
                self._draw_path()
                self._draw_trails()
        if slow:
            tip_err = onp.linalg.norm(s.tip - self.target) * 1000
            self.rcm_err_text.value = f"{s.rcm_error * 1000:.3f}"
            self.depth_text.value = f"{s.depth_past_rcm * 1000:.1f}"
            self.shaft_text.value = (
                ("OK" if s.insertion_ok else "OUT OF RANGE")
                + f" ({s.insertion * 1000:.0f} mm, allowed {S_MIN * 1000:.0f}-{S_MAX * 1000:.0f})"
            )
            self.tip_err_text.value = f"{tip_err:.3f}"
            ratio = self.joint_speed / self.kin.velocity_limit
            j = int(onp.argmax(ratio))
            self.jspeed_text.value = f"J{j + 1}: {onp.rad2deg(self.joint_speed[j]):.1f} deg/s ({ratio[j] * 100:.0f}%)"
            self.progress.value = 100.0 * self.u / max(1, len(self.path_pts) - 1)
            h = onp.asarray(self.history)
            self.err_plot.data = (h[:, 0], h[:, 1], h[:, 2])

    def _draw_path(self) -> None:
        pts = self.path_pts
        colors = (
            onp.tile(onp.array([80, 160, 255]), (len(pts), 1))
            if self.path_status is None or len(self.path_status) != len(pts)
            else self.path_status
        )
        seg = onp.stack([pts[:-1], pts[1:]], axis=1) + self.path_normals[:-1, None] * 0.0005
        seg_colors = onp.stack([colors[:-1], colors[1:]], axis=1).astype(onp.uint8)
        self.server.scene.add_line_segments("/path", seg, seg_colors, thickness=4.0, thickness_units="screen")

    def _draw_trails(self) -> None:
        for name, trail, color in (
            ("/trail_tip", self.trail_tip, (40, 90, 255)),
            ("/trail_rcm", self.trail_rcm, (230, 30, 30)),
        ):
            if len(trail) > 1:
                p = onp.asarray(trail)
                # Discard segments across jumps (e.g. wrap-around or path reset) > 25 mm
                diff = onp.linalg.norm(p[1:] - p[:-1], axis=1)
                valid = diff < 0.025
                if onp.any(valid):
                    seg = onp.stack([p[:-1][valid], p[1:][valid]], axis=1)
                    self.server.scene.add_line_segments(
                        name, seg, color, thickness=2.5, thickness_units="screen"
                    )
                else:
                    self.server.scene.remove_by_name(name)
            else:
                self.server.scene.remove_by_name(name)

    def run(self) -> None:
        logger.info("Ready: open the Viser URL above.")
        n = 0
        while True:
            start = time.perf_counter()
            self.tick()
            self.render(slow=n % 5 == 0)
            n += 1
            time.sleep(max(0.0, CONTROL_DT - (time.perf_counter() - start)))


def main(port: int = 8080) -> None:
    """OpenRCM: Remote Center of Motion motion-planning simulator (Dobot CR5 + instrument).

    Args:
        port: Viser web UI port.
    """
    RCMApp(port).run()


if __name__ == "__main__":
    tyro.cli(main)
