"""Focused exactness matrix for the automatically dispatched K2 V-split.

The cases are deliberately small (``H=2`` and at most 64 tokens) so this file
is suitable for the focused B300 test entry.  Coverage is nevertheless across
fixed and varlen dispatch, both cu_seqlens integer types, empty sequences, all
state input/output modes, and both supported state dtypes.  The automatic
dispatch selects V-split for these low-parallelism cases, and every result is
compared directly and bit-for-bit with ``torch_ref``.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F


if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

import flash_kda
from torch_ref import torch_ref


D = 128
H = 2
LOWER_BOUND = -5.0


@dataclasses.dataclass(frozen=True)
class StateCase:
    name: str
    has_input: bool
    has_output: bool
    dtype: torch.dtype | None


# ``no_state`` has no meaningful dtype.  Every state-bearing input/output
# mode is exercised independently with BF16 and FP32 state storage.
STATE_CASES = (
    StateCase("no_state", False, False, None),
    StateCase("in_only_bf16", True, False, torch.bfloat16),
    StateCase("in_only_fp32", True, False, torch.float32),
    StateCase("out_only_bf16", False, True, torch.bfloat16),
    StateCase("out_only_fp32", False, True, torch.float32),
    StateCase("in_out_bf16", True, True, torch.bfloat16),
    StateCase("in_out_fp32", True, True, torch.float32),
)


@dataclasses.dataclass(frozen=True)
class Problem:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    cu_seqlens: torch.Tensor | None
    num_sequences: int

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(D)


def _make_problem(
    *,
    batch: int,
    seq_len: int,
    cu_seqlens: torch.Tensor | None,
    num_sequences: int,
    seed: int,
) -> Problem:
    torch.manual_seed(seed)
    shape = (batch, seq_len, H, D)
    q = F.normalize(
        torch.randn(shape, dtype=torch.float32, device="cuda"), p=2, dim=-1
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn(shape, dtype=torch.float32, device="cuda"), p=2, dim=-1
    ).to(torch.bfloat16)
    return Problem(
        q=q,
        k=k,
        v=torch.randn(shape, dtype=torch.bfloat16, device="cuda"),
        g=torch.randn(shape, dtype=torch.bfloat16, device="cuda"),
        beta=torch.randn(shape[:-1], dtype=torch.bfloat16, device="cuda"),
        A_log=torch.rand(H, dtype=torch.float32, device="cuda"),
        dt_bias=torch.rand(H, D, dtype=torch.float32, device="cuda"),
        cu_seqlens=cu_seqlens,
        num_sequences=num_sequences,
    )


def _make_fixed_problem() -> Problem:
    # B>1 also checks fixed-length sequence-to-state indexing.  T=17 crosses a
    # chunk boundary and exercises the manual tail output store.
    return _make_problem(
        batch=2,
        seq_len=17,
        cu_seqlens=None,
        num_sequences=2,
        seed=101,
    )


def _make_stage_reuse_problem() -> Problem:
    # InputStages=3, so four full chunks are the smallest case that forces a
    # circular input stage to be reused.
    return _make_problem(
        batch=1,
        seq_len=64,
        cu_seqlens=None,
        num_sequences=1,
        seed=404,
    )


def _make_varlen_problem(cu_dtype: torch.dtype) -> Problem:
    # Repeated offsets put empty sequences at the beginning, middle, and end.
    # The two non-empty sequences cover both a one-token tail and 17 tokens
    # across two chunks without making the torch reference expensive.
    lengths = (0, 1, 0, 17, 0)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=cu_dtype, device="cuda")
    return _make_problem(
        batch=1,
        seq_len=offsets[-1],
        cu_seqlens=cu_seqlens,
        num_sequences=len(lengths),
        seed=202,
    )


def _state_template(problem: Problem, state_case: StateCase) -> torch.Tensor | None:
    if not state_case.has_input:
        return None
    assert state_case.dtype is not None
    # Generate through FP32 for deterministic BF16/FP32 cases with identical
    # logical values.  Each execution receives its own clone.
    torch.manual_seed(303)
    return torch.randn(
        (problem.num_sequences, H, D, D),
        dtype=torch.float32,
        device="cuda",
    ).to(state_case.dtype)


def _fresh_final_state(
    problem: Problem, state_case: StateCase
) -> torch.Tensor | None:
    if not state_case.has_output:
        return None
    assert state_case.dtype is not None
    # NaN makes a missing state store fail exact comparison, including for an
    # empty sequence whose expected state is the input state or all zeros.
    return torch.full(
        (problem.num_sequences, H, D, D),
        float("nan"),
        dtype=state_case.dtype,
        device="cuda",
    )


def _run_reference(
    problem: Problem,
    state_case: StateCase,
    initial_template: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    out = torch.full_like(problem.q, float("nan"))
    final_state = _fresh_final_state(problem, state_case)
    torch_ref(
        problem.q,
        problem.k,
        problem.v,
        problem.g,
        problem.beta,
        problem.scale,
        out,
        A_log=problem.A_log,
        dt_bias=problem.dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=(
            initial_template.clone() if initial_template is not None else None
        ),
        final_state=final_state,
        cu_seqlens=problem.cu_seqlens,
    )
    torch.cuda.synchronize()
    return out, final_state


def _run_kernel(
    problem: Problem,
    state_case: StateCase,
    initial_template: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    out = torch.full_like(problem.q, float("nan"))
    final_state = _fresh_final_state(problem, state_case)
    workspace = torch.empty(
        flash_kda.get_workspace_size(
            problem.q.shape[0] * problem.q.shape[1],
            H,
            problem.num_sequences,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    flash_kda.fwd(
        problem.q,
        problem.k,
        problem.v,
        problem.g,
        problem.beta,
        problem.scale,
        out,
        A_log=problem.A_log,
        dt_bias=problem.dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=(
            initial_template.clone() if initial_template is not None else None
        ),
        final_state=final_state,
        cu_seqlens=problem.cu_seqlens,
        workspace=workspace,
    )
    torch.cuda.synchronize()
    return out, final_state


def _assert_vsplit_exact(problem: Problem, state_case: StateCase) -> None:
    initial_template = _state_template(problem, state_case)
    expected_out, expected_state = _run_reference(
        problem, state_case, initial_template
    )

    actual_out, actual_state = _run_kernel(
        problem, state_case, initial_template
    )
    assert torch.equal(actual_out, expected_out), (
        f"output differs from torch_ref for {state_case.name}"
    )
    if expected_state is not None:
        assert actual_state is not None
        assert torch.equal(actual_state, expected_state), (
            f"final_state differs from torch_ref for {state_case.name}"
        )


@pytest.mark.parametrize("state_case", STATE_CASES, ids=lambda case: case.name)
def test_vsplit_fixed_exact(state_case: StateCase) -> None:
    _assert_vsplit_exact(_make_fixed_problem(), state_case)


def test_vsplit_stage_reuse_exact() -> None:
    _assert_vsplit_exact(
        _make_stage_reuse_problem(),
        StateCase("stage_reuse_fp32", True, True, torch.float32),
    )


@pytest.mark.parametrize(
    "cu_dtype",
    (torch.int32, torch.int64),
    ids=("cu_int32", "cu_int64"),
)
@pytest.mark.parametrize("state_case", STATE_CASES, ids=lambda case: case.name)
def test_vsplit_varlen_exact(
    state_case: StateCase, cu_dtype: torch.dtype
) -> None:
    _assert_vsplit_exact(_make_varlen_problem(cu_dtype), state_case)
