import math

import pytest
import torch

from mondeq_solver import ConvergenceError, MonDEQSolver, SolveResult


def model_tensors(dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
    identity = torch.eye(2, dtype=dtype)
    return {
        "H1": identity,
        "B1": identity,
        "H2": identity,
        "B2": identity,
        "C1": identity,
        "D1": identity,
        "u_ext": -torch.ones(2, 1, dtype=dtype),
    }


def test_diagonal_model_converges_to_expected_fixed_point() -> None:
    solver = MonDEQSolver(alpha_base=0.1)
    result = solver.solve(
        **model_tensors(),
        sigma_val=1.5,
        target_tol=1e-12,
        return_info=True,
    )

    expected_u = torch.full((2, 1), 1.0 / 11.0, dtype=torch.float64)
    expected_v = torch.full((2, 1), 15.0 / 121.0, dtype=torch.float64)
    assert isinstance(result, SolveResult)
    assert result.converged
    assert result.iterations == 2
    assert result.residual_norm <= 1e-12
    assert torch.allclose(result.u, expected_u, atol=1e-12, rtol=0)
    assert torch.allclose(result.v, expected_v, atol=1e-12, rtol=0)


def test_default_return_value_remains_a_state_tuple() -> None:
    states = MonDEQSolver().solve(**model_tensors())

    assert isinstance(states, tuple)
    assert len(states) == 2


def test_one_dimensional_external_input_is_accepted() -> None:
    tensors = model_tensors()
    tensors["u_ext"] = tensors["u_ext"].squeeze(-1)

    result = MonDEQSolver().solve(**tensors, return_info=True)

    assert result.u.shape == (2, 1)
    assert result.v.shape == (2, 1)


def test_float32_inputs_are_converted_to_configured_dtype() -> None:
    result = MonDEQSolver(dtype=torch.float64).solve(
        **model_tensors(torch.float32),
        return_info=True,
    )

    assert result.u.dtype == torch.float64
    assert result.v.dtype == torch.float64


def test_nonconvergence_raises_with_diagnostics() -> None:
    with pytest.raises(ConvergenceError) as captured:
        MonDEQSolver().solve(**model_tensors(), max_iter=1, target_tol=1e-15)

    assert not captured.value.result.converged
    assert captured.value.result.iterations == 1
    assert math.isfinite(captured.value.result.residual_norm)


def test_nonconverged_result_can_be_returned_explicitly() -> None:
    result = MonDEQSolver().solve(
        **model_tensors(),
        max_iter=1,
        target_tol=1e-15,
        return_info=True,
        raise_on_nonconvergence=False,
    )

    assert not result.converged
    assert result.iterations == 1


@pytest.mark.parametrize("alpha_base", [0.0, -1.0, float("inf")])
def test_invalid_alpha_is_rejected(alpha_base: float) -> None:
    with pytest.raises(ValueError):
        MonDEQSolver(alpha_base=alpha_base)


def test_incompatible_shapes_are_rejected() -> None:
    tensors = model_tensors()
    tensors["C1"] = torch.eye(3, 2, dtype=torch.float64)

    with pytest.raises(ValueError, match="B2 has shape"):
        MonDEQSolver().solve(**tensors)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("target_tol", 0.0), ("target_tol", -1.0), ("max_iter", 0)],
)
def test_invalid_run_parameters_are_rejected(keyword: str, value: float) -> None:
    with pytest.raises(ValueError):
        MonDEQSolver().solve(**model_tensors(), **{keyword: value})
