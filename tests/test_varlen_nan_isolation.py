"""Sequences in one varlen call must not change each other's output or state.

For a sequence whose last chunk has ``n = len % 16`` valid rows, the kernels
also load beta and v for the ``16 - n`` rows after the sequence end: the next
sequence of the same head, or, for the last sequence of the call, the first
tokens of the next head. Those rows must not reach the valid rows. Multiplying
them by zero is not enough, because ``0 * NaN`` is NaN, so each case puts one
NaN row into a neighbour and checks that the victim stays finite.

Shapes follow Kimi-K3 at TP8 (12 heads, head dim 128), where this turned one
NaN request into many bad requests in the same batch.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

import flash_kda


H = 12
D = 128
LOWER_BOUND = -5.0


def _make_inputs(lens, state_dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    T, N = sum(lens), len(lens)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, device="cuda")

    x = {name: (rnd(1, T, H, D) * 0.5).to(torch.bfloat16) for name in ("q", "k", "v", "g")}
    for name in ("q", "k"):
        x[name] = F.normalize(x[name].float(), dim=-1).to(torch.bfloat16)
    x["beta"] = rnd(1, T, H).to(torch.bfloat16)
    x["A_log"] = (rnd(H) * 0.5).float()
    x["dt_bias"] = (rnd(H, D) * 0.1).float()
    x["init"] = (rnd(N, H, D, D) * 0.05).to(state_dtype)
    return x


def _run(x, lens, state_dtype):
    T, N = sum(lens), len(lens)
    offsets = [0]
    for length in lens:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    out = torch.empty((1, T, H, D), dtype=torch.bfloat16, device="cuda")
    final = torch.empty((N, H, D, D), dtype=state_dtype, device="cuda")
    flash_kda.fwd(
        x["q"], x["k"], x["v"], x["g"], x["beta"], D ** -0.5, out,
        x["A_log"], x["dt_bias"], LOWER_BOUND,
        initial_state=x["init"].clone(), final_state=final, cu_seqlens=cu_seqlens,
    )
    torch.cuda.synchronize()
    return out[0], final


def _many_sequences():
    # Enough sequences to exceed the V-split threshold on current GPUs, so the
    # non-V-split recurrence kernel runs too. The victim is followed by a
    # one-token sequence, then one-token fillers.
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    return [700, 1] + [1] * (2 * num_sms)


CASES = [
    # (lens, poisoned tensor, poisoned row, victim sequence)
    pytest.param([1, 700], "beta", 0, 1, id="last_seq_wraps_into_next_head_beta"),
    pytest.param([700, 1], "v", 700, 0, id="next_seq_v"),
    pytest.param([700, 1], "beta", 700, 0, id="next_seq_beta"),
    pytest.param([1, 704], "beta", 0, 1, id="full_last_chunk"),
    pytest.param(None, "beta", 700, 0, id="next_seq_beta_many_sequences"),
]


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("lens, tensor, row, victim", CASES)
def test_nan_in_neighbour_does_not_leak(lens, tensor, row, victim, state_dtype):
    lens = lens or _many_sequences()
    x = _make_inputs(lens, state_dtype)

    out, final = _run(x, lens, state_dtype)
    assert torch.isfinite(out).all() and torch.isfinite(final).all()

    x[tensor][0, row] = float("nan")  # one NaN row (all heads) in another sequence
    out, final = _run(x, lens, state_dtype)

    start = sum(lens[:victim])
    victim_out = out[start:start + lens[victim]]
    bad_rows = (~torch.isfinite(victim_out)).any(-1).any(-1).nonzero().flatten().tolist()
    assert not bad_rows, f"victim output rows {bad_rows} are not finite"
    assert torch.isfinite(final[victim]).all(), "victim final state is not finite"
