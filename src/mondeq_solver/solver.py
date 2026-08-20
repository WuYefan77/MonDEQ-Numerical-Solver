"""Numerical iteration for a two-block cascaded equilibrium model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import torch

TensorPair: TypeAlias = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class SolveResult:
    """States and numerical diagnostics from a solver run."""

    u: torch.Tensor
    v: torch.Tensor
    converged: bool
    iterations: int
    step_size: float
    step_norm: float
    residual_norm: float

    @property
    def states(self) -> TensorPair:
        """Return the equilibrium states as the historical ``(u, v)`` tuple."""

        return self.u, self.v


class ConvergenceError(RuntimeError):
    """Raised when the iteration reaches ``max_iter`` without converging."""

    def __init__(self, result: SolveResult):
        self.result = result
        super().__init__(
            "MonDEQ iteration did not converge within "
            f"{result.iterations} iterations "
            f"(step norm={result.step_norm:.3e}, "
            f"fixed-point residual={result.residual_norm:.3e})."
        )


class MonDEQSolver:
    """Solve the public repository's two-block projected fixed-point model.

    The implementation updates the first block, forms the forward coupling from
    that updated state, and then updates the second block. Its automatically
    selected step size is derived from the two uncoupled affine block
    operators. Convergence of a particular coupled problem still depends on
    the supplied matrices and parameters.
    """

    def __init__(
        self,
        alpha_base: float = 0.1,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if not isinstance(alpha_base, (int, float)) or isinstance(alpha_base, bool):
            raise TypeError("alpha_base must be a real scalar.")
        if not math.isfinite(float(alpha_base)) or alpha_base <= 0:
            raise ValueError("alpha_base must be finite and greater than zero.")

        try:
            dtype_probe = torch.empty((), dtype=dtype)
        except (TypeError, RuntimeError) as exc:
            raise TypeError(
                "dtype must be a valid floating-point torch dtype."
            ) from exc
        if not dtype_probe.is_floating_point():
            raise TypeError("dtype must be a floating-point torch dtype.")

        self.alpha = float(alpha_base)
        self.dtype = dtype

    def _compute_block_step_size(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
    ) -> float:
        """Return a step size derived from the two affine block operators.

        This calculation does not include the forward-coupling matrices and is
        therefore a numerical default, not a general convergence certificate
        for every coupled system.
        """

        identity_1 = torch.eye(h1.shape[0], dtype=self.dtype, device=h1.device)
        identity_2 = torch.eye(h2.shape[0], dtype=self.dtype, device=h2.device)
        operator_1 = identity_1 + self.alpha * h1
        operator_2 = identity_2 + self.alpha * h2

        lipschitz_bound = max(
            torch.linalg.matrix_norm(operator_1, ord=2).item(),
            torch.linalg.matrix_norm(operator_2, ord=2).item(),
        )
        if not math.isfinite(lipschitz_bound) or lipschitz_bound <= 0:
            raise ValueError("Could not derive a finite positive block step size.")
        return 1.0 / lipschitz_bound

    @staticmethod
    def _require_matrix(name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional tensor.")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a real floating-point dtype.")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain only finite values.")

    def _prepare_inputs(
        self,
        h1: torch.Tensor,
        b1: torch.Tensor,
        h2: torch.Tensor,
        b2: torch.Tensor,
        c1: torch.Tensor,
        d1: torch.Tensor,
        u_ext: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        matrices = {
            "H1": h1,
            "B1": b1,
            "H2": h2,
            "B2": b2,
            "C1": c1,
            "D1": d1,
        }
        for name, matrix in matrices.items():
            self._require_matrix(name, matrix)

        if not isinstance(u_ext, torch.Tensor):
            raise TypeError("u_ext must be a torch.Tensor.")
        if not u_ext.is_floating_point():
            raise TypeError("u_ext must have a real floating-point dtype.")
        if u_ext.ndim == 1:
            u_ext = u_ext.unsqueeze(-1)
        elif u_ext.ndim != 2 or u_ext.shape[1] != 1:
            raise ValueError("u_ext must be a vector or a single-column matrix.")
        if not torch.isfinite(u_ext).all().item():
            raise ValueError("u_ext must contain only finite values.")

        tensors = (*matrices.values(), u_ext)
        reference_device = h1.device
        if any(tensor.device != reference_device for tensor in tensors):
            raise ValueError("All input tensors must be on the same device.")

        n1 = h1.shape[0]
        n2 = h2.shape[0]
        input_size = u_ext.shape[0]
        coupling_size = c1.shape[0]

        expected_shapes = {
            "H1": (n1, n1),
            "B1": (n1, input_size),
            "H2": (n2, n2),
            "B2": (n2, coupling_size),
            "C1": (coupling_size, n1),
            "D1": (coupling_size, input_size),
        }
        for name, expected in expected_shapes.items():
            actual = matrices[name].shape
            if actual != expected:
                raise ValueError(
                    f"{name} has shape {tuple(actual)}; expected {expected}."
                )

        return tuple(tensor.to(dtype=self.dtype) for tensor in tensors)

    @staticmethod
    def _validate_run_parameters(
        sigma_val: float,
        target_tol: float,
        max_iter: int,
    ) -> None:
        for name, value in {"sigma_val": sigma_val, "target_tol": target_tol}.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a real scalar.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if target_tol <= 0:
            raise ValueError("target_tol must be greater than zero.")
        if not isinstance(max_iter, int) or isinstance(max_iter, bool):
            raise TypeError("max_iter must be an integer.")
        if max_iter <= 0:
            raise ValueError("max_iter must be greater than zero.")

    def _iteration_map(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        h1: torch.Tensor,
        b1: torch.Tensor,
        h2: torch.Tensor,
        b2: torch.Tensor,
        c1: torch.Tensor,
        d1: torch.Tensor,
        u_ext: torch.Tensor,
        sigma_val: float,
        step_size: float,
    ) -> TensorPair:
        identity_1 = torch.eye(h1.shape[0], dtype=self.dtype, device=h1.device)
        identity_2 = torch.eye(h2.shape[0], dtype=self.dtype, device=h2.device)

        target_1 = -self.alpha * (b1 @ u_ext)
        gradient_1 = (identity_1 + self.alpha * h1) @ u - target_1
        next_u = torch.relu(u - step_size * gradient_1)

        coupled_input = sigma_val * (c1 @ next_u + d1 @ u_ext)
        target_2 = -self.alpha * (b2 @ coupled_input)
        gradient_2 = (identity_2 + self.alpha * h2) @ v - target_2
        next_v = torch.relu(v - step_size * gradient_2)
        return next_u, next_v

    def solve(
        self,
        H1: torch.Tensor,
        B1: torch.Tensor,
        H2: torch.Tensor,
        B2: torch.Tensor,
        C1: torch.Tensor,
        D1: torch.Tensor,
        u_ext: torch.Tensor,
        sigma_val: float = 1.5,
        target_tol: float = 1e-10,
        max_iter: int = 10_000,
        *,
        return_info: bool = False,
        raise_on_nonconvergence: bool = True,
    ) -> TensorPair | SolveResult:
        """Run the two-block iteration.

        Args:
            H1, B1, H2, B2, C1, D1, u_ext: Model tensors. See the README
                for the required dimensions.
            sigma_val: Scalar applied to the forward-coupled input.
            target_tol: Stopping tolerance for the sum of consecutive-state
                Euclidean norms.
            max_iter: Maximum number of outer iterations.
            return_info: Return :class:`SolveResult` instead of ``(u, v)``.
            raise_on_nonconvergence: Raise :class:`ConvergenceError` when the
                iteration exhausts ``max_iter``.

        The fixed-point residual in :class:`SolveResult` is the distance
        between the returned state and one additional application of the
        implemented block-iteration map.
        """

        self._validate_run_parameters(sigma_val, target_tol, max_iter)
        h1, b1, h2, b2, c1, d1, u_ext = self._prepare_inputs(
            H1, B1, H2, B2, C1, D1, u_ext
        )
        step_size = self._compute_block_step_size(h1, h2)

        u = torch.zeros(h1.shape[0], 1, dtype=self.dtype, device=h1.device)
        v = torch.zeros(h2.shape[0], 1, dtype=self.dtype, device=h2.device)
        step_norm = math.inf
        converged = False

        for iteration in range(1, max_iter + 1):
            next_u, next_v = self._iteration_map(
                u,
                v,
                h1,
                b1,
                h2,
                b2,
                c1,
                d1,
                u_ext,
                float(sigma_val),
                step_size,
            )
            if (
                not torch.isfinite(next_u).all().item()
                or not torch.isfinite(next_v).all().item()
            ):
                raise FloatingPointError(
                    "The iteration produced a non-finite state. Check the model "
                    "matrices, coupling strength, and scaling."
                )

            step_norm = (
                torch.linalg.vector_norm(next_u - u)
                + torch.linalg.vector_norm(next_v - v)
            ).item()
            u, v = next_u, next_v
            if step_norm <= target_tol:
                converged = True
                break

        residual_u, residual_v = self._iteration_map(
            u,
            v,
            h1,
            b1,
            h2,
            b2,
            c1,
            d1,
            u_ext,
            float(sigma_val),
            step_size,
        )
        residual_norm = (
            torch.linalg.vector_norm(residual_u - u)
            + torch.linalg.vector_norm(residual_v - v)
        ).item()
        result = SolveResult(
            u=u,
            v=v,
            converged=converged,
            iterations=iteration,
            step_size=step_size,
            step_norm=step_norm,
            residual_norm=residual_norm,
        )

        if not converged and raise_on_nonconvergence:
            raise ConvergenceError(result)
        return result if return_info else result.states
