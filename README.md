![Tests](https://github.com/PlatypusBytes/solvers/actions/workflows/workflow.yml/badge.svg)
[![codecov](https://codecov.io/gh/PlatypusBytes/solvers/graph/badge.svg?token=CRWV3A3WLR)](https://codecov.io/gh/PlatypusBytes/solvers)

# PuggleSolvers

```text
     _______.  ______    __      ____    ____  _______ .______       _________.
    /       | /  __  \  |  |     \   \  /   / |   ____||   _  \     /         |
   |   (----`|  |  |  | |  |      \   \/   /  |  |__   |  |_)  |   |   (------`
    \   \    |  |  |  | |  |       \      /   |   __|  |      /     \   \
.----)   |   |  `--'  | |  `----.   \    /    |  |____ |  |\  \------)   |
|_______/     \______/  |_______|    \__/     |_______|| _| `.__________/
```

**PuggleSolvers** is a Python library of numerical time-integration and static solvers for dynamics problems described by the equation of motion:

$$M\ddot{u} + C\dot{u} + Ku = F(t)$$

where $M$ is the mass matrix, $C$ is the damping matrix, $K$ is the stiffness matrix, $u$ is the displacement vector, and $F(t)$ is the external force vector.

It targets finite-element workflows and supports both dense and sparse matrices, optional GPU acceleration via CuPy, and non-linear force callbacks.

## Solvers

### Static

| Class | Module |
|---|---|
| `StaticSolver` | `solvers.static_solver` |

Solves $Ku = F$ using an incremental formulation. Supports non-linear force callbacks.

### Dynamic (time-integration)

| Class | Module | Type | Mass matrix | Reference |
|---|---|---|---|---|
| `NewmarkExplicit` | `solvers.newmark_solver` | Explicit | Consistent | [Newmark (1959)](https://ascelibrary.org/doi/10.1061/JMCEA3.0000098) |
| `NewmarkImplicitForce` | `solvers.newmark_solver` | Implicit | Consistent | [Newmark (1959)](https://ascelibrary.org/doi/10.1061/JMCEA3.0000098) |
| `NewmarkExplicitGPU` | `solvers.newmark_solver` | Explicit (GPU) | Consistent | [Newmark (1959)](https://ascelibrary.org/doi/10.1061/JMCEA3.0000098) |
| `NewmarkImplicitForceGPU` | `solvers.newmark_solver` | Implicit (GPU) | Consistent | [Newmark (1959)](https://ascelibrary.org/doi/10.1061/JMCEA3.0000098) |
| `ZhaiSolver` | `solvers.zhai_solver` | Explicit | Consistent | [Zhai (1996)](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0207%2819961230%2939%3A24%3C4199%3A%3AAID-NME39%3E3.0.CO%3B2-Y) |
| `HHTImplicitForce` | `solvers.HHT_solver` | Implicit | Consistent | [Hilber (1977)](https://onlinelibrary.wiley.com/doi/10.1002/eqe.4290050306) |
| `CentralDifferenceSolver` | `solvers.central_difference_solver` | Explicit | Consistent / Lumped | [Bathe (1996)](https://web.mit.edu/kjb/www/Principal_Publications/FEP_Binder.2nd_Edition_7th_Printing%20_1-2021.hybrid.pdf) |
| `BatheSolver` | `solvers.bathe_solver` | Explicit | Consistent / Lumped | [Noh & Bathe (2013)](https://www.sciencedirect.com/science/article/abs/pii/S0045794913001934) |

The implicit Newmark and HHT solvers use a Newton–Raphson iteration to handle non-linear forces.

**Mass lumping** for `CentralDifferenceSolver` and `BatheSolver` is controlled via `LumpingMethod`:

| Value | Description |
|---|---|
| `LumpingMethod.DiagonalScaling` | Proportional diagonal scaling (default) |
| `LumpingMethod.RowSum` | Row-sum lumping |
| `LumpingMethod.NONE` | Consistent mass matrix (no lumping) |

Import with: `from solvers.utils import LumpingMethod`

**GPU solver** (`solvers.newmark_solver_gpu`) requires [CuPy](https://cupy.dev/).
To install with GPU support, see the [Installation](#installation) section below.
The GPU solver only supports the Conjugate Gradient linear solver and Jacobi preconditioning.


## Linear System Solvers

All solvers accept an optional `linear_solver` argument (subclass of `LinearSolversABC`):

| Class | Method | Notes |
|---|---|---|
| `SparseDirectSolverLU` | LU factorisation | Default; caches LU factorisation |
| `SparseDirectSolver` | LU factorisation | No caching |
| `DenseDirectSolver` | LU factorisation | Caches LU; dense matrices only |
| `CGSolver` | Conjugate Gradient | Iterative; supports preconditioner |
| `GMRESSolver` | GMRES | Iterative; supports preconditioner |
| `BICSTABSolver` | BiCGSTAB | Iterative; supports preconditioner |


## Preconditioners

Iterative solvers accept an optional `preconditioner` argument (subclass of `PreconditionerABC`):

| Class | Description |
|---|---|
| `JacobiPreconditioner` | Diagonal (Jacobi) preconditioner |
| `SSORPreconditioner` | Symmetric SOR; configurable `omega` (0 < ω < 2) |
| `ILUPreconditioner` | Incomplete LU factorisation |


## Quick Start

```python
import numpy as np
from scipy.sparse import csc_matrix
from solvers.newmark_solver import NewmarkExplicit

# 2-DOF system (example from Bathe §9.2.4)
M = csc_matrix(np.array([[2, 0], [0, 1]], dtype=float))
K = csc_matrix(np.array([[6, -2], [-2, 4]], dtype=float))
C = csc_matrix(np.zeros((2, 2)))

n_steps = 12
t_step = 0.28
time = np.linspace(0, n_steps * t_step, n_steps + 1)

# Force matrix: shape (n_dof, n_timesteps) — constant 10 N on DOF 1
F = csc_matrix(np.tile([[0], [10]], (1, len(time))))

solver = NewmarkExplicit()
solver.initialise(2, time)
solver.calculate(M, C, K, F, 0, n_steps)

print(solver.v)   # displacement history, shape (n_dof, n_timesteps)
```


## Installation

To install the latest stable release from PyPI:

```bash
pip install PuggleSolvers
```

Or install from source:

```bash
git clone https://github.com/PlatypusBytes/solvers.git
cd solvers
pip install .
```

For GPU support (optional), install the CuPy package that matches your local CUDA version.

```bash
pip install "cupy-cuda11x[ctk]"   # CUDA 11.x
pip install "cupy-cuda12x[ctk]"   # CUDA 12.x
pip install "cupy-cuda13x[ctk]"   # CUDA 13.x
```

To check your CUDA version, run `nvcc --version` or `nvidia-smi` in the terminal.

See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) to find the right package for your system.
The rest of the library works without CuPy.


## License

[MIT](LICENSE)
