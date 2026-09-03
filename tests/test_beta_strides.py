import math

import pytest
import torch

import flash_kda


def _run(q, k, v, g, beta, initial_state, A_log, dt_bias, cu_seqlens):
    output = torch.empty_like(q)
    final_state = torch.empty_like(initial_state)
    kwargs = {"cu_seqlens": cu_seqlens} if cu_seqlens is not None else {}
    flash_kda.fwd(
        q,
        k,
        v,
        g,
        beta,
        1.0 / math.sqrt(q.shape[-1]),
        output,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        initial_state=initial_state,
        final_state=final_state,
        **kwargs,
    )
    torch.cuda.synchronize()
    return output, final_state


@pytest.mark.parametrize("seq_lens", [None, [17, 20]])
def test_row_strided_beta_matches_contiguous(seq_lens):
    torch.manual_seed(0)
    batch = 2 if seq_lens is None else 1
    tokens = 35 if seq_lens is None else sum(seq_lens)
    sequences = batch if seq_lens is None else len(seq_lens)
    heads, dimension = 4, 128
    shape = (batch, tokens, heads, dimension)

    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    beta_storage = torch.randn(
        (batch, tokens, heads + 3), dtype=torch.bfloat16, device="cuda"
    )
    beta_strided = beta_storage[..., :heads]
    assert beta_strided.stride(1) == heads + 3
    assert not beta_strided.is_contiguous()

    initial_state = torch.randn(
        (sequences, heads, dimension, dimension),
        dtype=torch.bfloat16,
        device="cuda",
    )
    A_log = torch.rand(heads, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(
        (heads, dimension), dtype=torch.float32, device="cuda"
    )
    cu_seqlens = None
    if seq_lens is not None:
        cu_seqlens = torch.tensor(
            [0, seq_lens[0], sum(seq_lens)], dtype=torch.int32, device="cuda"
        )

    expected = _run(
        q,
        k,
        v,
        g,
        beta_strided.contiguous(),
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
    )
    actual = _run(
        q,
        k,
        v,
        g,
        beta_strided,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
    )

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
