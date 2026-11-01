"""Dimensions of the straight RCM instrument (laparoscopic-style), along the flange +z axis."""

ADAPTER_LENGTH = 0.05  # flange adapter (thick part), m
ADAPTER_RADIUS = 0.03
SHAFT_LENGTH = 0.30  # thin shaft, m
SHAFT_RADIUS = 0.004
TIP_LENGTH = 0.012  # conical tip
TOOL_LENGTH = ADAPTER_LENGTH + SHAFT_LENGTH + TIP_LENGTH  # flange -> tip

# The RCM must lie on the thin shaft, at least this far from either end of it.
INSERTION_MARGIN = 0.02
