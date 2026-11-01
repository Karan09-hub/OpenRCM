"""Benchmark RCM-constrained path following over every surface x path pattern.

Writes benchmarks/rcm_benchmark.md: per-case max / RMS RCM error, max tip error,
feasibility, IK time per sample, plus an RCM-off comparison.

Run:
    python tools/rcm_benchmark.py
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import jax
import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openrcm.kinematics import HOME_CFG  # noqa: E402
from openrcm.paths import BENCHMARK_FILE  # noqa: E402
from openrcm.rcm import RCMKinematics, plan_path  # noqa: E402
from openrcm.rcm_scene import PATTERNS, SURFACES, WorkObject  # noqa: E402

OUTPUT = BENCHMARK_FILE
PORT_HEIGHT = 0.08  # m above the path center, along the surface normal
TIP_SPEED = 0.03  # m/s


def main() -> None:
    kin = RCMKinematics()
    obj = WorkObject()
    kin.solve(obj.path_world()[0][0], HOME_CFG, obj.point_world(0, 0, PORT_HEIGHT))  # JIT compile
    rows, all_rcm, all_ms = [], [], []
    t_start = time.time()
    for surface in SURFACES:
        for pattern in PATTERNS:
            obj = WorkObject(shape=surface)
            obj.pattern = pattern
            pts, nls = obj.path_world()
            port = obj.point_world(0, 0, PORT_HEIGHT)
            rep = plan_path(kin, pts, nls, HOME_CFG, port, enforce_rcm=True, tip_speed=TIP_SPEED)

            rcm_max_mm = rep.rcm_error.max() * 1000
            rcm_rms_mm = float(onp.sqrt((rep.rcm_error**2).mean())) * 1000
            tip_max_mm = rep.tip_error.max() * 1000
            ms_sample = rep.ik_seconds / len(pts) * 1000
            speed_pct = rep.speed_ratio.max() * 100
            margin_deg = rep.limit_margin_deg.min()

            status = "PASS" if rep.feasible else "FAIL"
            rows.append(
                f"| {surface:<12} | {pattern:<10} | {len(pts):>5} | {rcm_max_mm:>7.3f} | "
                f"{rcm_rms_mm:>7.3f} | {tip_max_mm:>7.3f} | {ms_sample:>6.2f} | "
                f"{speed_pct:>5.1f}% | {margin_deg:>6.1f}° | {status} |"
            )
            all_rcm.append(rcm_max_mm)
            all_ms.append(ms_sample)

    # RCM OFF comparison on Dome + Circle
    obj = WorkObject(shape="Dome")
    obj.pattern = "Circle"
    pts, nls = obj.path_world()
    port = obj.point_world(0, 0, PORT_HEIGHT)
    rep_off = plan_path(kin, pts, nls, HOME_CFG, port, enforce_rcm=False, tip_speed=TIP_SPEED)
    off_max_mm = rep_off.rcm_error.max() * 1000

    header = f"""# OpenRCM Benchmark Results

Generated on **{time.strftime("%Y-%m-%d %H:%M:%S")}**
System: **{platform.processor() or platform.machine()}**, Python {platform.python_version()}, JAX {jax.__version__}

- Incision port: **{PORT_HEIGHT * 1000:.0f} mm** above surface center along surface normal
- Commanded tip speed: **{TIP_SPEED * 100:.0f} cm/s**, path sample spacing: **2 mm**
- Feasibility criterion: RCM error < 0.5 mm, tip error < 1.0 mm, all joints within limits & speeds

| Surface | Pattern | Samples | Max RCM (mm) | RMS RCM (mm) | Max Tip (mm) | IK ms/samp | Max speed | Min margin | Status |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---:|
"""
    summary = f"""
## Summary
- **Cases tested:** {len(rows)} ({len(SURFACES)} surfaces x {len(PATTERNS)} patterns)
- **All cases feasible:** {"YES" if all(r.endswith("PASS |") for r in rows) else "NO"}
- **Peak RCM error across all cases:** {max(all_rcm):.3f} mm (enforced) vs **{off_max_mm:.1f} mm** (RCM disabled)
- **Mean IK solve time:** {onp.mean(all_ms):.2f} ms per sample ({1000 / onp.mean(all_ms):.0f} Hz)
- **Total benchmark runtime:** {time.time() - t_start:.1f} s
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(header + "\n".join(rows) + "\n" + summary)
    print(f"Wrote benchmark results to {OUTPUT}")


if __name__ == "__main__":
    main()
