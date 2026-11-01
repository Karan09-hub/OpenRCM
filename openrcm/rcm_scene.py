"""The work object and the paths drawn on it, for the RCM simulator.

The object has a top surface z = h(x, y) in its own frame, over a square footprint.
A 2D pattern (circle, square, ...) in the object's xy-plane is projected onto that
surface, so the path follows the object wherever it is moved or rotated.
"""

from __future__ import annotations

import jaxlie
import numpy as onp
import trimesh

FOOTPRINT = 0.24  # object top is FOOTPRINT x FOOTPRINT (m)
BASE_DEPTH = 0.06  # solid block under the surface (display only)

SURFACES = ("Flat pad", "Dome", "Wavy tissue")
PATTERNS = ("Circle", "Square", "Figure-8", "Spiral", "Zig-zag", "Line")


def surface_height(shape: str, x: onp.ndarray, y: onp.ndarray) -> onp.ndarray:
    if shape == "Dome":
        # Spherical cap: 4 cm high over a 22 cm wide circle.
        a, h = 0.11, 0.04
        radius = (a**2 + h**2) / (2 * h)
        r2 = onp.minimum(x**2 + y**2, a**2)
        return onp.sqrt(radius**2 - r2) - (radius - h)
    if shape == "Wavy tissue":
        return 0.012 * onp.sin(2 * onp.pi * x / 0.12) * onp.cos(2 * onp.pi * y / 0.16)
    return onp.zeros_like(x)


def surface_normal(shape: str, x: onp.ndarray, y: onp.ndarray, eps: float = 1e-4) -> onp.ndarray:
    dhdx = (surface_height(shape, x + eps, y) - surface_height(shape, x - eps, y)) / (2 * eps)
    dhdy = (surface_height(shape, x, y + eps) - surface_height(shape, x, y - eps)) / (2 * eps)
    n = onp.stack([-dhdx, -dhdy, onp.ones_like(x)], axis=-1)
    return n / onp.linalg.norm(n, axis=-1, keepdims=True)


def pattern_xy(pattern: str, size: float, n: int = 400) -> onp.ndarray:
    """2D pattern centered at the origin, `size` = overall width (m)."""
    t = onp.linspace(0.0, 1.0, n)
    r = size / 2
    if pattern == "Circle":
        a = 2 * onp.pi * t
        xy = onp.stack([r * onp.cos(a), r * onp.sin(a)], axis=1)
    elif pattern == "Square":
        corners = onp.array([[r, r], [-r, r], [-r, -r], [r, -r], [r, r]])
        xy = _polyline(corners, n)
    elif pattern == "Figure-8":
        a = 2 * onp.pi * t
        xy = onp.stack([r * onp.sin(a), r * onp.sin(a) * onp.cos(a)], axis=1)
    elif pattern == "Spiral":
        a = 6 * onp.pi * t
        xy = onp.stack([r * t * onp.cos(a), r * t * onp.sin(a)], axis=1)
    elif pattern == "Zig-zag":
        rows = 5
        pts = []
        for i, yv in enumerate(onp.linspace(-r, r, rows)):
            xs = (-r, r) if i % 2 == 0 else (r, -r)
            pts += [(xs[0], yv), (xs[1], yv)]
        xy = _polyline(onp.array(pts), n)
    else:  # Line
        xy = onp.stack([onp.linspace(-r, r, n), onp.zeros(n)], axis=1)
    return _resample(xy, spacing=0.002)


def _polyline(corners: onp.ndarray, n: int) -> onp.ndarray:
    seg = onp.linalg.norm(onp.diff(corners, axis=0), axis=1)
    s = onp.concatenate([[0], onp.cumsum(seg)])
    u = onp.linspace(0, s[-1], n)
    return onp.stack([onp.interp(u, s, corners[:, 0]), onp.interp(u, s, corners[:, 1])], axis=1)


def _resample(xy: onp.ndarray, spacing: float) -> onp.ndarray:
    """Even spacing along the curve, so a constant index rate means constant speed."""
    seg = onp.linalg.norm(onp.diff(xy, axis=0), axis=1)
    s = onp.concatenate([[0], onp.cumsum(seg)])
    u = onp.linspace(0, s[-1], max(2, int(s[-1] / spacing) + 1))
    return onp.stack([onp.interp(u, s, xy[:, 0]), onp.interp(u, s, xy[:, 1])], axis=1)


class WorkObject:
    """Surface shape + pose + path pattern. Everything returned is in world coordinates."""

    def __init__(self, position=(-0.42, 0.0, 0.10), shape: str = "Flat pad") -> None:
        self.pose = jaxlie.SE3.from_translation(onp.asarray(position, dtype=onp.float64))
        self.shape = shape
        self.pattern = "Circle"
        self.size = 0.10
        self.path_offset = onp.zeros(2)  # pattern center on the surface (object xy)

    # --------------------------------------------------------------- geometry

    def mesh_local(self) -> trimesh.Trimesh:
        """Top surface as a height-field grid plus a block underneath (object frame)."""
        n = 61
        g = onp.linspace(-FOOTPRINT / 2, FOOTPRINT / 2, n)
        X, Y = onp.meshgrid(g, g, indexing="ij")
        Z = surface_height(self.shape, X, Y)
        top = onp.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        faces = []
        for i in range(n - 1):
            for j in range(n - 1):
                a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
                faces += [(a, b, c), (a, c, d)]
        surface = trimesh.Trimesh(top, onp.array(faces), process=False)
        bottom = float(Z.min()) - BASE_DEPTH
        block = trimesh.creation.box(extents=(FOOTPRINT, FOOTPRINT, Z.min() - bottom))
        block.apply_translation((0, 0, (Z.min() + bottom) / 2 - 0.0005))
        return trimesh.util.concatenate([surface, block])

    def path_local(self) -> tuple[onp.ndarray, onp.ndarray]:
        xy = pattern_xy(self.pattern, self.size) + self.path_offset
        xy = onp.clip(xy, -FOOTPRINT / 2, FOOTPRINT / 2)
        z = surface_height(self.shape, xy[:, 0], xy[:, 1])
        return onp.column_stack([xy, z]), surface_normal(self.shape, xy[:, 0], xy[:, 1])

    def path_world(self) -> tuple[onp.ndarray, onp.ndarray]:
        """Path points and surface normals in world coordinates."""
        pts, normals = self.path_local()
        R = onp.asarray(self.pose.rotation().as_matrix())
        return pts @ R.T + onp.asarray(self.pose.translation()), normals @ R.T

    def point_world(self, x: float, y: float, height: float = 0.0) -> onp.ndarray:
        """Surface point at object (x, y), raised by `height` along the surface normal."""
        local = onp.array([x, y, float(surface_height(self.shape, onp.array(x), onp.array(y)))])
        local = local + height * surface_normal(self.shape, onp.array(x), onp.array(y))
        return onp.asarray(self.pose @ local)

    def path_center_xy(self) -> onp.ndarray:
        return self.path_offset.copy()

    # ----------------------------------------------------------------- picking

    def raycast(self, origin: onp.ndarray, direction: onp.ndarray) -> onp.ndarray | None:
        """First hit of a world ray with the top surface, in object (x, y); None if missed."""
        inv = self.pose.inverse()
        o = onp.asarray(inv @ onp.asarray(origin))
        d = onp.asarray(inv.rotation() @ onp.asarray(direction))
        t = onp.linspace(0.0, 5.0, 5001)
        p = o + t[:, None] * d
        inside = (onp.abs(p[:, 0]) <= FOOTPRINT / 2) & (onp.abs(p[:, 1]) <= FOOTPRINT / 2)
        above = p[:, 2] - surface_height(self.shape, p[:, 0], p[:, 1]) > 0
        crossing = onp.nonzero(inside[1:] & above[:-1] & ~above[1:])[0]
        if len(crossing) == 0:
            return None
        lo, hi = t[crossing[0]], t[crossing[0] + 1]
        for _ in range(30):  # bisection on the crossing
            mid = (lo + hi) / 2
            q = o + mid * d
            if q[2] > surface_height(self.shape, q[0], q[1]):
                lo = mid
            else:
                hi = mid
        q = o + lo * d
        return q[:2]
