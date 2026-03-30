import torch
from typing import Tuple, List, Optional

class MonDEQSolver:
    """
    High-precision numerical solver for Monotone Equilibrium Networks (MonDEQs).
    Implements the proposed Two-Loop Splitting (k=1) algorithm to achieve 
    accelerated 'anytime convergence' and machine-limit precision.
    """
    
    def __init__(self, alpha_base: float = 0.1, dtype: torch.dtype = torch.float64):
        """
        Initializes the solver with numerical precision settings.
        
        Args:
            alpha_base (float): Base step size parameter for the resolvent.
            dtype (torch.dtype): Tensor data type, defaults to float64 for high precision (10^-14).
        """
        self.alpha = alpha_base
        self.dtype = dtype
        self.lr_safe = None

    def _compute_safe_learning_rate(self, H1: torch.Tensor, H2: torch.Tensor) -> float:
        """Calculates the theoretical safe step size based on Lipschitz continuity."""
        N = H1.shape[0]
        I = torch.eye(N, dtype=self.dtype, device=H1.device)
        L1 = torch.linalg.matrix_norm(I + self.alpha * H1, ord=2)
        L2 = torch.linalg.matrix_norm(I + self.alpha * H2, ord=2)
        return 1.0 / max(L1.item(), L2.item())

    def solve_splitting_k1(
        self,
        H1: torch.Tensor, B1: torch.Tensor,
        H2: torch.Tensor, B2: torch.Tensor,
        C1: torch.Tensor, D1: torch.Tensor,
        u_ext: torch.Tensor,
        sigma_val: float = 1.5,
        target_tol: float = 1e-10,
        max_iter: int = 5000,
        z_truth: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """
        Executes the Two-Loop Splitting (k=1) algorithm.
        
        Args:
            [... Matrix inputs omitted for brevity in docstring ...]
            target_tol (float): Target L2 norm global error.
            max_iter (int): Maximum number of allowed iterations.
            z_truth (torch.Tensor, optional): Ground truth tensor for real-time error tracking.
            
        Returns:
            Tuple containing the final states (u, v) and the convergence history.
        """
        N = H1.shape[0]
        device = H1.device
        
        # Initialize safe learning rate dynamically
        self.lr_safe = self._compute_safe_learning_rate(H1, H2)
        
        # Initialize states
        u = torch.zeros(N, 1, dtype=self.dtype, device=device)
        v = torch.zeros(N, 1, dtype=self.dtype, device=device)
        I = torch.eye(N, dtype=self.dtype, device=device)
        
        # Pre-compute operators to save FLOPs in the inner loop
        op_H1 = I + self.alpha * H1
        op_H2 = I + self.alpha * H2
        
        history = []
        
        for _ in range(max_iter):
            # Block 1: Splitting update
            u = torch.relu(u - self.lr_safe * (op_H1 @ u - (u - self.alpha * B1 @ u_ext)))
            
            # Forward coupling
            u2_in = sigma_val * (C1 @ u + D1 @ u_ext)
            
            # Block 2: Splitting update
            v = torch.relu(v - self.lr_safe * (op_H2 @ v - (v - self.alpha * B2 @ u2_in)))
            
            # Track error if ground truth is provided (for benchmarking)
            if z_truth is not None:
                current_state = torch.cat((u, v))
                err = torch.norm(current_state - z_truth).item()
                history.append(err)
                if err < target_tol:
                    break
                    
        return u, v, history
