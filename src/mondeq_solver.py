import torch
from typing import Tuple

class MonDEQSolver:
    """
    Industrial-grade numerical solver for Monotone Equilibrium Networks (MonDEQs).
    Implements the core Two-Loop Splitting (k=1) algorithm for extreme efficiency.
    """
    
    def __init__(self, alpha_base: float = 0.1, dtype: torch.dtype = torch.float64):
        """
        Initializes the mathematical engine.
        Float64 is set by default to ensure machine-limit precision.
        """
        self.alpha = alpha_base
        self.dtype = dtype

    def _compute_safe_lr(self, H1: torch.Tensor, H2: torch.Tensor) -> float:
        """Dynamically calculates the safe Lipschitz-based step size."""
        N1, N2 = H1.shape[0], H2.shape[0]
        I1 = torch.eye(N1, dtype=self.dtype, device=H1.device)
        I2 = torch.eye(N2, dtype=self.dtype, device=H2.device)
        
        # Calculate maximum eigenvalue bounds
        L = max(torch.linalg.matrix_norm(I1 + self.alpha * H1, ord=2).item(),
                torch.linalg.matrix_norm(I2 + self.alpha * H2, ord=2).item())
        return 1.0 / L

    def solve(
        self,
        H1: torch.Tensor, B1: torch.Tensor,
        H2: torch.Tensor, B2: torch.Tensor,
        C1: torch.Tensor, D1: torch.Tensor,
        u_ext: torch.Tensor,
        sigma_val: float = 1.5,
        target_tol: float = 1e-10,
        max_iter: int = 10000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes the pure Two-Loop Splitting algorithm without any benchmarking overhead.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The converged equilibrium states (u, v).
        """
        device = H1.device
        N1, N2 = H1.shape[0], H2.shape[0]
        
        # 1. Initialize safe learning rate and states
        lr_safe = self._compute_safe_lr(H1, H2)
        u = torch.zeros(N1, 1, dtype=self.dtype, device=device)
        v = torch.zeros(N2, 1, dtype=self.dtype, device=device)
        
        I1 = torch.eye(N1, dtype=self.dtype, device=device)
        I2 = torch.eye(N2, dtype=self.dtype, device=device)
        
        # 2. Pre-compute constant target terms to save FLOPs
        target_f1 = -self.alpha * (B1 @ u_ext)
        
        # 3. Core Splitting Loop
        for _ in range(max_iter):
            u_prev, v_prev = u.clone(), v.clone()
            
            # --- Block 1 Splitting Step ---
            grad_u = (I1 + self.alpha * H1) @ u - target_f1
            u = torch.relu(u - lr_safe * grad_u)
            
            # --- Forward Coupling ---
            u2_in = sigma_val * (C1 @ u + D1 @ u_ext)
            target_f2 = -self.alpha * (B2 @ u2_in)
            
            # --- Block 2 Splitting Step ---
            grad_v = (I2 + self.alpha * H2) @ v - target_f2
            v = torch.relu(v - lr_safe * grad_v)
            
            # --- Check Internal Convergence ---
            # Instead of comparing to a 'ground truth', a real solver checks 
            # if the internal states have stopped moving.
            diff = torch.norm(u - u_prev) + torch.norm(v - v_prev)
            if diff < target_tol:
                break
                
        return u, v
