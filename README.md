# High-Efficiency Solvers for Cascaded Monotone Operator Equilibrium Networks (MonDEQs)

This repository provides an industrial-grade, PyTorch-based numerical solver for Deep Analog Neural Networks (specifically, Cascaded MonDEQs). 

By leveraging **Operator Splitting Theory (Eckstein 1989)**, we replace traditional sequential solving methods with a simultaneous **Two-Loop Splitting** architecture. This mathematical framework guarantees global convergence while achieving machine-level precision ($10^{-14}$).

## 🚀 Key Innovations

### 1. Eliminating the Computational "Dead Zone" (Anytime Convergence)
Traditional sequential solvers suffer from a massive latency period where downstream layers wait idly for upstream layers to converge. Our splitting algorithm enables **immediate, simultaneous updates** across all layers.
As shown below, our method achieves $10^{-5}$ precision while the baseline is still at $10^0$, making it ideal for low-latency neuromorphic hardware.

![Anytime Convergence](assets/convergence_analysis.pdf) 
*(Replace with your actual relative path)*

### 2. The Power of Inexact Splitting ($1588\times$ Speedup)
A key theoretical discovery is that **inner-loop convergence is unnecessary**. By relaxing the internal accuracy to a single proximal step ($k=1$), we reduce the total computational operations by a factor of **1588x** without sacrificing the final $10^{-14}$ accuracy. Global synchronization proves vastly superior to local precision.

![1588x Efficiency](assets/speedup_1588x_benchmark.pdf)
*(Replace with your actual relative path)*

## 💻 The Core Solver

The solver is designed for stability under strong coupling regimes ($\sigma \ge 1.5$) where standard global solvers often lose monotonicity and fail. It dynamically computes a safe Lipschitz-based step size to prevent numerical explosion.

```python
# A minimal example of the solver's API
from solver import MonDEQSolver

# Initialize the engine (Float64 is recommended for machine-limit precision)
solver = MonDEQSolver(alpha_base=0.1)

# Execute the Two-Loop Splitting (k=1) algorithm
# H1, H2: Resistive networks (Strongly Monotone)
# B, C, D: Coupling and input matrices
u_state, v_state = solver.solve(
    H1, B1, H2, B2, C1, D1, u_ext, 
    sigma_val=1.5, 
    target_tol=1e-10
)
