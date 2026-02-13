# Wave Equation Solver

Group information:
- member-1: Luca, Spreafico, 10660926
- member-2: Marta, Arzeni, 107 98912
- member-3: Matilde, Colombo, 107 82110
- member-4: Roberto Manea, 10813429

A finite element solver for the 2D wave equation using the [deal.II](https://www.dealii.org/) library. The project implements two time-stepping schemes (**Newmark-β** and **θ-method**) with both serial and MPI-parallel versions.

## Mathematical Problem

We solve the wave equation on a square domain Ω = [-1, 1]²:

```
∂²u/∂t² = Δu + f(x,y,t)    in Ω × (0, T]
u = 0                       on ∂Ω (Dirichlet BC)
u(x,y,0) = u₀(x,y)         (initial displacement)
∂u/∂t(x,y,0) = v₀(x,y)     (initial velocity)
```

The spatial discretization uses Q1 (bilinear) finite elements on a structured quadrilateral mesh.

### Test Cases

Two manufactured solutions are provided to verify convergence:

| Case | Exact Solution | Forcing Term | Description |
|------|----------------|--------------|-------------|
| **EX1** | `sin(π(x+1)/2) · sin(π(y+1)/2) · cos(t)` | `(π²/2 - 1) · φ(x,y) · cos(t)` | Forced vibration |
| **EX2** | `sin(π(x+1)/2) · sin(π(y+1)/2) · cos(π/√2 · t)` | `0` | Free vibration (homogeneous) |

Both cases use the same initial condition: `u₀ = sin(π(x+1)/2) · sin(π(y+1)/2)` and `v₀ = 0`.

## Project Structure

```
├── src/
│   ├── common/
│   │   └── WaveTestCases.hpp     # Shared test cases, exact solutions, forcing terms
│   ├── Newmark/                  # Newmark-β method (MPI parallel)
│   │   ├── Wave.cpp/.hpp
│   │   └── main.cpp
│   └── Theta_Method/
│       ├── serial/               # θ-method (sequential)
│       │   ├── Wave.cpp/.hpp
│       │   └── main.cpp
│       └── parallel/             # θ-method (MPI parallel)
│           ├── WaveParallel.cpp/.hpp
│           └── main.cpp
├── params_newmark.prm            # Parameter file for Newmark solver
├── params_theta.prm              # Parameter file for θ-method solvers
├── src/plot_energy.py            # Energy evolution plotting
├── src/dispersion.py             # Dispersion analysis plotting
├── common/cmake-common.cmake
└── CMakeLists.txt
```

## Time Discretization Methods

### Newmark-β Method
A direct second-order formulation that advances displacement, velocity, and acceleration simultaneously. It solves one linear system per time step and is energy-conserving for the default parameters (β = 0.25, γ = 0.5).

### θ-Method
Reformulates the wave equation as a first-order system in (u, v). With θ = 0.5 (Crank-Nicolson), the scheme is second-order accurate in time. It solves two linear systems per time step (one for u, one for v).

## Requirements

- **deal.II** ≥ 9.3.1 with MPI and Trilinos enabled
- **CMake** ≥ 3.12
- **MPI** implementation (OpenMPI, MPICH, etc.)
- **Boost** ≥ 1.72
- **Python 3** with NumPy and Matplotlib (optional, for plotting)

## Build Instructions

```bash
mkdir build && cd build
cmake ..
make
```

This produces three executables: `Wave_Newmark`, `Wave_Serial`, and `Wave_Parallel`.

## Running Simulations

All executables accept an optional `.prm` parameter file as their first argument. If no file is provided, built-in defaults are used.

**Serial execution:**
```bash
./Wave_Serial                        # uses defaults
./Wave_Serial ../params_theta.prm    # reads from parameter file
```

**Parallel execution (e.g., 4 processes):**
```bash
mpirun -np 4 ./Wave_Newmark ../params_newmark.prm
mpirun -np 4 ./Wave_Parallel ../params_theta.prm
```

## Configuration

Simulation parameters are configured through `.prm` files using `dealii::ParameterHandler`. Two example files are provided at the project root.

**`params_newmark.prm`:**
```
set Test case    = EX2
set Refinement   = 7
set Degree       = 1
set Final time   = 2.0
set Time step    = 0.01
set Beta         = 0.25
set Gamma        = 0.5
set Domain left  = -1.0
set Domain right = 1.0
```

**`params_theta.prm`** (shared by serial and parallel):
```
set Test case    = EX2
set Refinement   = 7
set Degree       = 1
set Final time   = 2.0
set Time step    = 0.01
set Theta        = 0.5
set Domain left  = -1.0
set Domain right = 1.0
```

### Parameter Reference

| Parameter | Description | Default | Applies to |
|-----------|-------------|---------|------------|
| `Test case` | `EX1` (forced) or `EX2` (free vibration) | `EX2` | All |
| `Refinement` | Number of global mesh refinements | `7` | All |
| `Degree` | FE polynomial degree | `1` | All |
| `Final time` | Simulation end time T | `2.0` | All |
| `Time step` | Time step size Δt | `0.01` | All |
| `Beta` | Newmark β parameter | `0.25` | Newmark |
| `Gamma` | Newmark γ parameter | `0.5` | Newmark |
| `Theta` | θ-method parameter (0.5 = Crank-Nicolson) | `0.5` | θ-method |
| `Domain left` | Left boundary of square domain | `-1.0` | All |
| `Domain right` | Right boundary of square domain | `1.0` | All |

## Output Files

| File | Content |
|------|---------|
| `output-newmark-*.vtu` / `.pvtu` | Newmark solution snapshots for ParaView |
| `solution-*.vtu` / `.pvtu` | θ-method solution snapshots for ParaView |
| `errors.csv` / `errors_parallel.csv` | L² and H¹ error norms vs. time |
| `energy.csv` / `energy_parallel.csv` | Total, kinetic, and potential energy vs. time |
| `center_point_solution.csv` | Newmark solution at (0,0) over time |
| `center_point_solution_theta.csv` | θ-method solution at (0,0) over time |

### Energy Computation
The discrete energy is computed as:
```
E(t) = ½ vᵀMv + ½ uᵀAu
```
where M is the mass matrix, A is the stiffness matrix, and (u, v) are the displacement and velocity vectors.

## Visualization

**Plot error evolution:**
```bash
python3 src/dispersion.py center_point_solution.csv
```

**Compare solutions across mesh refinements:**
```bash
python3 src/dispersion.py mesh_4.csv mesh_5.csv mesh_6.csv mesh_7.csv
```

**Compare energy for different Newmark parameters:**
```bash
python3 src/plot_energy.py energy_0.5_0.25.csv energy_0.6_0.3025.csv
```

**View solution in ParaView:**
Open the `.pvtu` files to visualize wave propagation and mesh partitioning.

## Implementation Details

The project uses several deal.II utilities for clean MPI-aware code:

- **`ConditionalOStream`** for rank-0-only console output without manual rank checks.
- **`TimerOutput`** for MPI-aggregated wall-time profiling with automatic summary tables.
- **`ParameterHandler`** for runtime configuration via `.prm` files with built-in validation.

The parallel implementations use `parallel::fullydistributed::Triangulation` for mesh distribution, Trilinos sparse matrices and vectors, and CG solvers with identity or SSOR preconditioners.

## References

- deal.II Step-23: Wave equation tutorial
- Hughes, T.J.R. *The Finite Element Method* — Newmark method derivation
- Quarteroni, A. *Numerical Models for Differential Problems* — θ-method analysis