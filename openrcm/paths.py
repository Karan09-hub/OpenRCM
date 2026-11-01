"""OpenRCM project file locations."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
XACRO = ROOT / "Assets" / "cr5_robot.xacro"
DAE_DIR = ROOT / "Assets"
URDF = ROOT / "urdf" / "cr5_rcm.urdf"
RCM_URDF = URDF
OBJ_DIR = ROOT / "meshes" / "obj"
BENCHMARK_FILE = ROOT / "benchmarks" / "rcm_benchmark.md"
