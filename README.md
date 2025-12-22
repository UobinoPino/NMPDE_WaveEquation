# Wave Equation Solver

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
│   ├── Newmark/              # Newmark-β method (MPI parallel)
│   │   ├── Wave.cpp/.hpp
│   │   └── Wave_Newmark.cpp  # Main driver
│   └── Theta_Method/
│       ├── serial/           # θ-method (sequential)
│       │   ├── Wave.cpp/.hpp
│       │   └── main.cpp
│       └── parallel/         # θ-method (MPI parallel)
│           ├── WaveParallel.cpp/.hpp
│           └── main.cpp
├── plot_energy.py            # Energy evolution plotting
├── plot_error.py             # Error norms plotting
└── CMakeLists.txt
```

## Time Discretization Methods

### Newmark-β Method
A direct second-order formulation that advances displacement, velocity, and acceleration simultaneously:
- **Parameters**: β = 0.25, γ = 0.5 (average acceleration, unconditionally stable)
- Solves one linear system per time step
- Energy-conserving for the chosen parameters

### θ-Method
Reformulates the wave equation as a first-order system in (u, v):
- **Parameter**: θ = 0.5 (Crank-Nicolson scheme)
- Solves two linear systems per time step (one for u, one for v)
- Second-order accurate in time

## Requirements

- **deal.II** ≥ 9.4 with MPI and Trilinos enabled
- **CMake** ≥ 3.13
- **MPI** implementation (OpenMPI, MPICH, etc.)
- **Python 3** with NumPy and Matplotlib (optional, for plotting)

## Build Instructions

```bash
mkdir build && cd build
cmake ..
make
```

This produces three executables:
- `wave_serial` — Serial θ-method
- `wave_parallel` — Parallel θ-method
- `wave_newmark` — Parallel Newmark-β method

## Running Simulations

**Serial execution:**
```bash
./wave_serial
```

**Parallel execution (e.g., 4 processes):**
```bash
mpirun -np 4 ./wave_newmark
mpirun -np 4 ./wave_parallel
```

## Configuration

To change the test case, edit the main file:
```cpp
const Wave::TestCase test_case = Wave::EX1;  // or Wave::EX2
```

### Simulation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `degree` | FE polynomial degree | 1 |
| `T` | Final simulation time | 2.0 |
| `delta_t` | Time step size | 0.01 |
| `n_refine` | Mesh refinement level | 7 |
| `beta` | Newmark parameter β | 0.25 |
| `gamma` | Newmark parameter γ | 0.5 |
| `theta` | θ-method parameter | 0.5 |
| `domain_left/right` | Domain bounds | -1.0 / 1.0 |

## Output Files

| File | Content |
|------|---------|
| `solution-*.vtu` / `solution.pvtu` | Solution snapshots for ParaView |
| `errors.csv` | L² and H¹ error norms vs. time |
| `energy.csv` | Total, kinetic, and potential energy vs. time |

### Energy Computation
The discrete energy is computed as:
```
E(t) = ½ vᵀMv + ½ uᵀAu
```
where M is the mass matrix, A is the stiffness matrix, and (u, v) are the displacement and velocity vectors.

## Visualization

**Plot error evolution:**
```bash
python3 plot_error.py build/errors.csv
```

**Compare energy for different parameters:**
```bash
python3 plot_energy.py energy_0.5_0.25.csv energy_0.6_0.3025.csv
```

**View solution in ParaView:**
Open the `.pvtu` files to visualize the wave propagation and mesh partitioning.

## Performance

The parallel implementations use:
- `parallel::fullydistributed::Triangulation` for mesh distribution
- Trilinos sparse matrices and vectors
- CG solver with SSOR preconditioner

Scalability results are printed in the format:
```
SCALABILITY_RESULT,<method>,<nprocs>,<time>
```

## References

- deal.II Step-23: Wave equation tutorial
- Hughes, T.J.R. *The Finite Element Method* — Newmark method derivation
- Quarteroni, A. *Numerical Models for Differential Problems* — θ-method analysis