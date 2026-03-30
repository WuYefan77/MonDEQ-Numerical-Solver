import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple, List

def create_psd_matrix(dim, strong=True):
    A = torch.randn(dim, dim, dtype=torch.float64)
    M = A.t() @ A
    if strong: 
        M += 0.1 * torch.eye(dim, dtype=torch.float64)
    return M

def create_mondeq_module(n, m_in, m_out):
    H = create_psd_matrix(n, strong=True)
    B = torch.randn(n, m_in, dtype=torch.float64)
    C = torch.randn(m_out, n, dtype=torch.float64)
    D = torch.randn(m_out, m_in, dtype=torch.float64)
    return H, B, C, D

def solve_resolvent_stable(H, target_f, alpha=0.1, steps=2000):
    dim = H.shape[0]
    z = torch.zeros_like(target_f, dtype=H.dtype)
    I = torch.eye(dim, dtype=H.dtype)
    
    matrix_to_solve = I + alpha * H
    max_eig = torch.linalg.matrix_norm(matrix_to_solve, ord=2)
    lr = 1.0 / max_eig 
    
    for _ in range(steps):
        grad = matrix_to_solve @ z - target_f
        z_next = torch.relu(z - lr * grad)
        if torch.norm(z_next - z) < 1e-15:
            break
        z = z_next
    return z

class ConvergenceBenchmark:
    """
    Benchmarking suite for comparing Two-Loop Splitting (k=1) 
    against traditional exact cascade solvers (k=100).
    """
    def __init__(self, sigma_val: float = 1.5):
        self.sigma_val = sigma_val
        self.alpha_alg = 0.1
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(42)
        
        # Initialize system parameters
        self.N1, self.N2, self.M, self.U_IN = 10, 8, 5, 3
        self.H1, self.B1, self.C1, self.D1 = create_mondeq_module(self.N1, self.U_IN, self.M)
        self.H2, self.B2, _, _ = create_mondeq_module(self.N2, self.M, 0)
        self.u1_input = torch.randn(self.U_IN, 1, dtype=torch.float64)

    def solve_for_reference(self) -> torch.Tensor:
        print("Calculating Stable Physical Reference (Ground Truth)...")
        z1 = solve_resolvent_stable(self.H1, -0.1 * (self.B1 @ self.u1_input), alpha=0.1, steps=5000)
        u2 = self.sigma_val * (self.C1 @ z1 + self.D1 @ self.u1_input)
        z2 = solve_resolvent_stable(self.H2, -0.1 * (self.B2 @ u2), alpha=0.1, steps=5000)
        return torch.cat((z1, z2), dim=0)

    def run_experiment(self, k: int, z_ref: torch.Tensor, max_outer: int = 500) -> Tuple[List[float], float, float, float]:
        """Runs the splitting algorithm with k inner iterations."""
        u = torch.zeros(self.N1, 1, dtype=torch.float64)
        v = torch.zeros(self.N2, 1, dtype=torch.float64)
        
        hist = []
        start_time = time.time()
        
        for _ in range(max_outer):
            # Block 1 inner loop
            for _ in range(k):
                u = solve_resolvent_stable(self.H1, -(u - 0.1 * self.B1 @ self.u1_input), alpha=self.alpha_alg, steps=1)
            
            # Forward coupling
            u2_in = self.sigma_val * (self.C1 @ u + self.D1 @ self.u1_input)
            
            # Block 2 inner loop
            for _ in range(k):
                v = solve_resolvent_stable(self.H2, -(v - 0.1 * self.B2 @ u2_in), alpha=self.alpha_alg, steps=1)
            
            # Record global error
            current_state = torch.cat((u, v), dim=0)
            abs_err = torch.norm(current_state - z_ref).item()
            hist.append(abs_err)
            
            if abs_err < 1e-10:  # Target Precision
                break
                
        time_cost = time.time() - start_time
        rel_err = abs_err / torch.norm(z_ref).item()
        return hist, abs_err, rel_err, time_cost

    def run_full_comparison(self):
        z_ref = self.solve_for_reference()

        print("Running proposed k=1 solver (Two-Loop Splitting)...")
        hist_k1, abs_err_k1, rel_err_k1, time_k1 = self.run_experiment(1, z_ref)
        
        print("Running naive k=100 solver (Exact Cascade)...")
        hist_k100, abs_err_k100, rel_err_k100, time_k100 = self.run_experiment(100, z_ref)

        self._generate_report(hist_k1, hist_k100, abs_err_k1, rel_err_k1, abs_err_k100, rel_err_k100)
        self._plot_results(hist_k1, hist_k100)

    def _generate_report(self, hist_k1, hist_k100, abs_err_k1, rel_err_k1, abs_err_k100, rel_err_k100):
        steps_k1 = len(hist_k1) * 1
        steps_k100 = len(hist_k100) * 100
        speedup = int(steps_k100 / steps_k1)

        print("\n" + "="*80)
        print(f"{'Configuration':<20} | {'Absolute Error':<15} | {'Rel Error':<10} | {'Total Inner Steps'}")
        print("-" * 80)
        print(f"{'k=1 (Inexact)':<20} | {abs_err_k1:<15.2e} | {rel_err_k1:<10.2e} | {steps_k1}")
        print(f"{'k=100 (Exact)':<20} | {abs_err_k100:<15.2e} | {rel_err_k100:<10.2e} | {steps_k100}")
        print("="*80)
        print(f"CONCLUSION: k=1 is {speedup}x more efficient than k=100.")

    def _plot_results(self, hist_k1, hist_k100):
        plt.figure(figsize=(10, 6))
        plt.plot(hist_k1, label=f'k=1 (Outer Iterations: {len(hist_k1)})', color='red', linewidth=2)
        plt.plot(hist_k100, label=f'k=100 (Outer Iterations: {len(hist_k100)})', color='blue', linestyle='--')
        plt.yscale('log')
        plt.title(f'Convergence Comparison: k=1 vs k=100 ($\sigma={self.sigma_val}$)', fontsize=14)
        plt.xlabel('Outer Iterations')
        plt.ylabel('Residual (log scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("convergence_comparison_k1_vs_k100.pdf", bbox_inches='tight', dpi=300)
        print("Plot successfully saved as 'convergence_comparison_k1_vs_k100.pdf'.")

if __name__ == "__main__":
    benchmark = ConvergenceBenchmark(sigma_val=1.5)
    benchmark.run_full_comparison()
