# Stefan Lake Freeze

A finite volume method (FVM) solver for simulating water freezing dynamics in lakes, implementing the classic Stefan problem with heat transfer and phase change modeling.

## Project Overview

This repository contains a comprehensive numerical simulation of ice formation on lake surfaces using the Finite Volume Method. The project models the thermodynamic processes involved in water freezing, including:

- **Heat diffusion** through ice and water layers
- **Phase change dynamics** at the ice-water interface
- **Convection effects** with velocity fields
- **Interface tracking** for ice-water boundaries

This was developed as a final project for the CTFD (Computational Thermal Fluid Dynamics) course.

## Key Features

- **FVM Solver**: Robust finite volume discretization for transient heat conduction
- **Stefan Problem Implementation**: Accurate phase change modeling with moving boundaries
- **Convection-Diffusion Coupling**: Integration of velocity fields with heat transport
- **Visualization Tools**: Animation and plotting utilities for simulation results
- **Comprehensive Testing Suite**: Multiple test cases validating solver accuracy

## Project Structure

```
Scripts/
  ├── fvm_solver.py          # Core FVM solver implementation
  ├── stefan_simulation.py    # Main Stefan problem simulation
  ├── velocity_field.py       # Velocity field generation and utilities
  ├── fl_field.py             # Freezing level field management
  └── __init__.py

Tests/
  ├── stefanTemperature.py    # Temperature profile validation
  ├── stefanEnergy.py         # Energy balance verification
  ├── convectionDiffusion.py  # Convection-diffusion coupling tests
  ├── interfaceTracking.py    # Ice-water interface tracking tests
  ├── gridindependence.py     # Grid convergence studies
  └── ...                      # Additional validation tests

Notebooks/
  └── FVM.ipynb               # Interactive simulation and analysis

Plots/                         # Generated figures and visualizations

LaTex/                         # Project documentation and reports
  ├── Outline/
  └── Report/
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
pip install -r requirements.txt
```

### Dependencies
- **numpy**: Numerical computations
- **matplotlib**: Visualization and animation
- **scipy**: Sparse linear algebra solvers

## Usage

### Running Simulations

```python
from Scripts.stefan_simulation import StefanSimulation

# Initialize and run simulation
sim = StefanSimulation()
sim.run()
sim.visualize()
```

### Running Tests

```bash
python Tests/stefanTemperature.py
python Tests/stefanEnergy.py
python Tests/interfaceTracking.py
```

### Interactive Notebook

Open the Jupyter notebook for interactive exploration:

```bash
jupyter notebook Notebooks/FVM.ipynb
```

## Mathematical Background

### The Stefan Problem

The Stefan problem describes heat diffusion with a moving boundary (phase change interface):

$$\frac{\partial T}{\partial t} = \alpha \nabla^2 T$$

with a moving interface condition:

$$L\frac{ds}{dt} = -k\nabla T|_{\text{interface}}$$

where:
- $T$ is temperature
- $\alpha$ is thermal diffusivity
- $s$ is interface position
- $L$ is latent heat of fusion
- $k$ is thermal conductivity

## Methodology

The project employs:
- **Finite Volume Method (FVM)**: Conservative discretization on unstructured grids
- **Implicit time stepping**: Unconditionally stable scheme
- **Sparse linear solvers**: Efficient solution of large systems
- **Interface tracking**: Accurate moving boundary representation

## Output

The simulation generates:
- Temperature field evolution over time
- Ice-water interface position tracking
- Energy balance diagnostics
- Animated visualizations of the freezing process

## Author

Developed as part of the CTFD course final project.

## References

See `LaTex/Report/references.bib` for detailed references on the Stefan problem and numerical methods. 