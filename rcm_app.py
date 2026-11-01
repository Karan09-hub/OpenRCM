"""Compatibility wrapper for running OpenRCM via `python rcm_app.py`."""

import tyro

from app import main

if __name__ == "__main__":
    tyro.cli(main)
