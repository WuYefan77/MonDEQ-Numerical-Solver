"""Minimal deterministic example for the public solver API."""

import torch

from mondeq_solver import MonDEQSolver


def main() -> None:
    dtype = torch.float64
    identity = torch.eye(2, dtype=dtype)
    external_input = -torch.ones(2, 1, dtype=dtype)

    solver = MonDEQSolver(alpha_base=0.1, dtype=dtype)
    result = solver.solve(
        H1=identity,
        B1=identity,
        H2=identity,
        B2=identity,
        C1=identity,
        D1=identity,
        u_ext=external_input,
        sigma_val=1.5,
        target_tol=1e-12,
        return_info=True,
    )

    print(f"converged: {result.converged}")
    print(f"iterations: {result.iterations}")
    print(f"fixed-point residual: {result.residual_norm:.3e}")
    print("u:", result.u.squeeze().tolist())
    print("v:", result.v.squeeze().tolist())


if __name__ == "__main__":
    main()
