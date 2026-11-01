"""Kinematics configuration for Dobot CR5 + RCM instrument."""

import numpy as onp

# Elbow-up "ready" pose (radians): J1..J6.
HOME_CFG = onp.deg2rad([0.0, 0.0, 90.0, 0.0, -90.0, 0.0])
