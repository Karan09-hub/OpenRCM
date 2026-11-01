# OpenRCM: Remote Center of Motion Simulation & Kinematics Framework

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-Accelerated%20IK-crimson.svg)](https://github.com/google/jax)
[![Viser](https://img.shields.io/badge/Viser-Interactive%203D%20UI-purple.svg)](https://github.com/nerfstudio-project/viser)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen.svg)](tests/)

**OpenRCM** is a simulation-first robotics framework for **Remote Center of Motion (RCM)** constrained kinematics, trajectory optimization, and visualization. Designed for surgical robotics and minimally invasive procedures (laparoscopy / keyhole surgery), OpenRCM solves constrained inverse kinematics in real time on a 6-DOF robot arm (**Dobot CR5**) manipulating a laparoscopic instrument through an incision fulcrum.

---

## Table of Contents
- [Why RCM?](#why-rcm)
- [Mathematical Formulation](#mathematical-formulation)
  - [Degrees of Freedom & Redundancy](#degrees-of-freedom--redundancy)
  - [Constraint Residuals](#constraint-residuals)
  - [Constrained Nonlinear Optimization](#constrained-nonlinear-optimization)
- [Key Features](#key-features)
- [Interactive Browser GUI](#interactive-browser-gui)
- [Benchmark Results](#benchmark-results)
- [Project Layout](#project-layout)
- [Installation & Quickstart](#installation--quickstart)
- [Running Tests & Benchmarks](#running-tests--benchmarks)
- [Future Roadmap](#future-roadmap)
- [License](#license)

---

## Why RCM?

In minimally invasive surgery (MIS), surgical instruments enter the patient's body cavity through small trocar ports (typically 5–12 mm). The abdominal or thoracic wall acts as a mechanical fulcrum:
- The instrument shaft **must pivot strictly about this fixed point in space**.
- Lateral or shearing forces at the incision cause severe tissue trauma, tearing, or port displacement.
- Meanwhile, the instrument tip must precisely follow complex operational trajectories (suturing, resection, cautery) on non-planar anatomical surfaces.

```
                  Robot Flange
                       │
                       ▼
               ┌────────────────┐
               │ Adapter (50mm) │
               └───────┬────────┘
                       │
                       │ Shaft (300mm, Ø8mm)
                       │
        ═══════════════╪═══════════════  ◄── Incision / Trocar Fulcrum (RCM Point)
          Body Wall    │                   (Zero lateral translation permitted)
                       │
                       ▼
                    Tip (12mm)  ────────►  Tracks 3D trajectory on tissue surface
```

OpenRCM lets you interactively examine this mechanism, evaluate constrained inverse kinematics under arbitrary port positions, and observe what happens when RCM constraints are violated.

---

## Mathematical Formulation

### Degrees of Freedom & Redundancy

A standard 6-axis manipulator has $6$ joint degrees of freedom ($q_1 \dots q_6$):
- **Tip Cartesian Position:** 3 constraints ($x, y, z$).
- **RCM Fulcrum Constraint:** 2 equality constraints (the line of the tool shaft must intersect the 3D incision point).
- **Total Task Constraints:** $3 + 2 = 5$ DOF.

Because $6 - 5 = 1$, the robot possesses **1 redundant degree of freedom**: rotation (roll) of the instrument around its own longitudinal shaft axis. OpenRCM exploits this redundancy using a regularization rest cost toward the previous configuration, guaranteeing smooth motion and eliminating null-space drift.

### Constraint Residuals

Let:
- $\mathbf{p} \in \mathbb{R}^3$ be the origin of the instrument frame (at the robot flange).
- $\mathbf{d} \in \mathbb{R}^3$ be the unit vector pointing along the instrument shaft ($+z$).
- $\mathbf{c} \in \mathbb{R}^3$ be the 3D world position of the RCM incision port.

The displacement vector is $\mathbf{v} = \mathbf{c} - \mathbf{p}$. The scalar insertion depth along the shaft is:
$$s = \mathbf{v} \cdot \mathbf{d}$$

The lateral RCM error (orthogonal distance between the incision fulcrum and shaft axis) is:
$$e_{\text{rcm}} = \| \mathbf{v} \times \mathbf{d} \|$$

In the tool's local frame:
$$\mathbf{d}_{\text{local}} = \mathbf{R}^T (\mathbf{c} - \mathbf{p})$$
The equality constraint is simply:
$$\begin{bmatrix} d_{\text{local}, x} \\ d_{\text{local}, y} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

Furthermore, the incision point must remain within the physical cylindrical shaft span (insertion bounds):
$$s_{\min} \le s \le s_{\max}$$

### Constrained Nonlinear Optimization

At every trajectory sample, OpenRCM formulates and solves a nonlinear least-squares problem with hard equality and inequality constraints using [JAXLS](https://github.com/brentyi/jaxls) and [JAX](https://github.com/google/jax):

$$\min_{\mathbf{q}} \quad 100 \, \| \mathbf{p}_{\text{tip}}(\mathbf{q}) - \mathbf{p}_{\text{target}} \|^2 + 0.02 \, \| \mathbf{q} - \mathbf{q}_{\text{prev}} \|^2$$

$$\begin{aligned}
\text{subject to} \quad & \mathbf{d}_{\text{local}, xy}(\mathbf{q}) = \mathbf{0} & \text{(RCM equality)} \\
& s_{\min} \le s(\mathbf{q}) \le s_{\max} & \text{(Insertion limits)} \\
& \mathbf{q}_{\min} \le \mathbf{q} \le \mathbf{q}_{\max} & \text{(Joint position limits)}
\end{aligned}$$

---

## Key Features

- **Sub-Millimeter RCM Precision:** Max RCM deviation under 0.15 mm (RMS < 0.04 mm) across all anatomical test surfaces and patterns.
- **Fast JAX-Compiled Solver:** Solves constrained IK at **> 200 Hz** on CPU, allowing real-time trajectory streaming and live interaction.
- **Surface & Path Geometry Generator:**
  - **Surfaces:** Flat surgical pad, Anatomical Dome (spherical cap simulating organ surface), and sinusoidal Wavy tissue.
  - **Patterns:** Circle, Square, Figure-8, Archimedean Spiral, Zig-zag raster, and Line.
- **RCM-On vs. RCM-Off Comparative Mode:**
  - **Enforce RCM = True:** Robot pivots strictly around the incision fulcrum; RCM error remains sub-millimeter.
  - **Enforce RCM = False ("Naive" Controller):** The robot maintains end-effector orientation relative to surface normals. The RCM trail visualizes massive lateral shaft displacement (> 30 mm), demonstrating the severe tissue laceration a naive controller would inflict.
- **Interactive Viser 3D UI:**
  - Live 3D robot articulation and shaft axis projection.
  - Real-time raycasting: click anywhere on the 3D tissue surface to reposition the incision port.
  - Interactive 6-DOF transform handles for the tissue object and port.
  - Real-time telemetry: live UPlot charts for joint angles, RCM error, and tip error.
- **Offline Trajectory Validation:** Pre-flight path feasibility check with color-coded safety rating:
  - 🟢 **Green:** Fully feasible within kinematics and velocity envelopes.
  - 🟠 **Orange:** Approaching joint limits or velocity thresholds (> 80%).
  - 🔴 **Red:** Infeasible (RCM constraint violation, joint limit breach, or branch jump).

---

## Interactive Browser GUI

Start the interactive visualizer:

```bash
python app.py
```

Then open `http://localhost:8080` in any web browser.

### Interactive Controls
| Control | Description |
|---|---|
| **Enforce RCM** | Checkbox to toggle RCM constraints on or off. Observe the red incision drag trail when toggled off. |
| **Port Height Slider** | Adjust the incision port height along the surface normal (20 mm to 200 mm). |
| **Place RCM: click object** | Raycasts into the 3D scene; click any point on the organ surface to set the incision fulcrum above it. |
| **Surface Dropdown** | Switch between `Flat pad`, `Dome`, and `Wavy tissue`. |
| **Pattern Dropdown** | Switch between `Circle`, `Square`, `Figure-8`, `Spiral`, `Zig-zag`, and `Line`. |
| **Check whole path** | Computes full trajectory offline, displays joint angle plots, and highlights infeasible points. |
| **Play / Pause / Speed** | Control trajectory replay speed and animation. |

---

## Benchmark Results

The benchmark systematically evaluates 18 surface-path combinations (3 surfaces $\times$ 6 trajectory patterns):

| Surface | Pattern | Samples | Max RCM (mm) | RMS RCM (mm) | Max Tip (mm) | IK ms/sample | Status |
|:---|:---|---:|---:|---:|---:|---:|:---:|
| Flat pad | Circle | 158 | 0.052 | 0.015 | 0.407 | 4.1 | PASS |
| Flat pad | Square | 201 | 0.144 | 0.026 | 0.812 | 4.0 | PASS |
| Flat pad | Figure-8 | 196 | 0.067 | 0.023 | 0.490 | 4.2 | PASS |
| Flat pad | Spiral | 240 | 0.048 | 0.020 | 0.523 | 4.3 | PASS |
| Flat pad | Zig-zag | 201 | 0.128 | 0.023 | 0.771 | 4.2 | PASS |
| Flat pad | Line | 51 | 0.055 | 0.021 | 0.380 | 4.0 | PASS |
| Dome | Circle | 158 | 0.057 | 0.018 | 0.419 | 4.2 | PASS |
| Dome | Square | 201 | 0.125 | 0.028 | 0.795 | 4.2 | PASS |
| Dome | Figure-8 | 196 | 0.076 | 0.024 | 0.504 | 4.1 | PASS |
| Dome | Spiral | 240 | 0.053 | 0.022 | 0.536 | 4.3 | PASS |
| Dome | Zig-zag | 201 | 0.134 | 0.026 | 0.782 | 4.2 | PASS |
| Dome | Line | 51 | 0.062 | 0.023 | 0.401 | 4.1 | PASS |
| Wavy tissue | Circle | 158 | 0.054 | 0.017 | 0.411 | 4.3 | PASS |
| Wavy tissue | Square | 201 | 0.131 | 0.027 | 0.801 | 4.1 | PASS |
| Wavy tissue | Figure-8 | 196 | 0.071 | 0.023 | 0.497 | 4.2 | PASS |
| Wavy tissue | Spiral | 240 | 0.051 | 0.021 | 0.531 | 4.2 | PASS |
| Wavy tissue | Zig-zag | 201 | 0.130 | 0.025 | 0.778 | 4.2 | PASS |
| Wavy tissue | Line | 51 | 0.058 | 0.022 | 0.389 | 4.1 | PASS |

> **Comparative Summary:**
> - **100% Feasibility:** 18/18 test cases fully feasible.
> - **Peak RCM Error (Enforced):** **0.144 mm**
> - **Peak Incision Drag (RCM Disabled):** **34.2 mm** (237x larger, demonstrating severe tissue trauma).
> - **Mean IK Solve Rate:** **~240 Hz** (4.1 ms per sample).

---

## Project Layout

```text
OpenRCM/
├── app.py                      # Main application entry point (Viser browser UI)
├── rcm_app.py                  # CLI alias for app.py
├── requirements.txt            # Python dependencies
├── OpenRCM_PLAN.md             # Theoretical project vision & research notes
├── LICENSE                     # MIT license
├── openrcm/                    # Core Python package
│   ├── __init__.py             # Exports public classes & constants
│   ├── env_check.py            # Environment & dependency validation
│   ├── paths.py                # Asset and model path definitions
│   ├── instrument.py           # Laparoscopic instrument dimensions
│   ├── kinematics.py           # Joint configuration presets
│   ├── rcm.py                  # Constrained IK solver, residuals & path reporting
│   └── rcm_scene.py            # Surface shapes, 2D/3D patterns & raycasting
├── tools/
│   ├── build_urdf.py           # Generates cr5_rcm.urdf and instrument meshes
│   └── rcm_benchmark.py        # Automated benchmark across all surfaces & patterns
├── benchmarks/
│   └── rcm_benchmark.md        # Full benchmark log
├── tests/
│   └── test_rcm.py             # Pytest test suite for geometry, IK, and limits
├── Assets/                     # Dobot CR5 robot DAE visual models & xacro
├── meshes/obj/                 # Instrument STL models (adapter, shaft, tip)
└── urdf/
    └── cr5_rcm.urdf            # Complete robot + instrument URDF
```

---

## Installation & Quickstart

### Prerequisites
- Linux / macOS / Windows (WSL2 recommended for Windows)
- Python 3.10, 3.11, or 3.12

### Setup

```bash
# 1. Clone repository
git clone https://github.com/your-username/OpenRCM.git
cd OpenRCM

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch visualizer
python app.py
```

---

## Running Tests & Benchmarks

### Unit Tests
Verify mathematical residuals, IK convergence, joint limits, and insertion checks:

```bash
pytest tests -v
```

### Benchmarks
Run the full 18-case matrix and generate benchmark documentation:

```bash
python tools/rcm_benchmark.py
```

---

## Future Roadmap

- [ ] Synthetic eye-in-hand and eye-to-hand calibration pipelines.
- [ ] Evaluation of calibration estimation error on downstream RCM violation.
- [ ] Articulated wristed instruments (e.g. EndoWrist 7-DOF kinematics).
- [ ] Dynamic trocar elasticity modeling and virtual fixture haptic feedback.

---

## License

This project is licensed under the [MIT License](LICENSE).
Dobot CR5 robot description assets are property of Dobot.
