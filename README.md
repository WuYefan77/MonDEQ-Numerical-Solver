# MonDEQ Numerical Solver: Two-Loop Splitting 🚀

A high-precision and ultrafast numerical solver for Monotone Equilibrium Networks (MonDEQs). By introducing a novel **Two-Loop Splitting ($k=1$)** algorithm, this solver achieves **machine-limit precision ($10^{-14}$)** and a **1588x computational speedup** compared to traditional sequential methods.

## 📊 Performance & Anytime Convergence

Unlike naive cascade solvers that fail to reduce global error until individual blocks converge, our inexact splitting mechanism ensures monotonic error decay from the very first iteration.

* 📉 **Convergence Analysis**: [View Anytime Convergence Plot](./assets/convergence_analysis.pdf)
* 🏎️ **Speedup Metrics**: [View 1588x Benchmark Report](./assets/speedup_1588x_benchmark.pdf)
* 🎓 **Theoretical Framework**: [View Research Poster  ](./assets/research_poster.pdf)

*(Note: The above PDF links contain detailed visual benchmarks and theoretical proofs.)*

## ⚙️ Mathematical Engine

The core logic, implemented in `src/mondeq_solver.py`, completely decouples the block updates. By utilizing analytical resolvents, the step updates for the latent states $u$ and $v$ are computed as follows:

$$u^{(k+1)} = \mathrm{ReLU} \left( u^{(k)} - \eta_{safe} \left( (I + \alpha H_1)u^{(k)} - (u^{(k)} - \alpha B_1 u_{ext}) \right) \right)$$

By limiting the inner iterative loop to a single step ($k=1$), we bypass the dense computational bottleneck of exact block solutions while strictly preserving the global fixed-point equivalence. 

## 💻 Quick Start (API Usage)

We provide a highly decoupled, OOP-based solver engine that is ready for industrial deployment.

```python
import torch
from src.mondeq_solver import MonDEQSolver

# 1. Initialize the high-precision solver (Float64 for 10^-14 precision)
solver = MonDEQSolver(alpha_base=0.1, dtype=torch.float64)

# 2. Mock some system matrices (Example dimensions: N=10, U_IN=3)
# (In practice, replace these with your actual MonDEQ weights)
H1, B1, H2, B2 = torch.eye(10), torch.randn(10, 3), torch.eye(8), torch.randn(8, 5)
C1, D1 = torch.randn(5, 10), torch.randn(5, 3)
u_ext = torch.randn(3, 1)

# 3. Execute the pure Two-Loop Splitting algorithm
u, v = solver.solve(
    H1, B1, H2, B2, C1, D1, u_ext,
    sigma_val=1.5,
    target_tol=1e-10,
    max_iter=5000
)

print(f"Convergence achieved! Latent state u shape: {u.shape}, v shape: {v.shape}")
