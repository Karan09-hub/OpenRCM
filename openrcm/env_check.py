"""Fail early with a clear message when OpenRCM is started without required packages."""

import importlib.util
import sys
from pathlib import Path

REQUIRED = ("jax", "jaxlie", "jaxls", "pyroki", "viser", "yourdfpy", "tyro")


def require_venv() -> None:
    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    if missing:
        venv_python = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
        script = Path(sys.argv[0]).name
        sys.exit(
            f"Missing packages: {', '.join(missing)}.\n"
            f"This interpreter ({sys.executable}) does not have the required dependencies. Run:\n"
            f"    {venv_python} {script}\n"
            f"Or install dependencies with: pip install -r requirements.txt"
        )
