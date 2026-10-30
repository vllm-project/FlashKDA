"""Full correctness tests for FlashKDA forward pass.

Compares cutlass kernel vs torch_ref for exact match across:
  - state_in / state_out: 4 combinations of None/present
  - state dtype: bf16, fp32
  - various H values (up to 256)
  - various sequence lengths (up to 1M, fixed-length and varlen)
  - D=128 (fixed)

Run with: pytest tests/test_fwd_full.py -x -v
Parallel:  pytest tests/test_fwd_full.py -x -v -n auto  (requires pytest-xdist)
"""

import torch
import torch.nn.functional as F
import math
import pytest

import flash_kda
from torch_ref import torch_ref

D = 128
LOWER_BOUND = -5.0


def _make_inputs(T, H, device="cuda"):
    torch.manual_seed(42)
    q = F.normalize(torch.randn((1, T, H, D), dtype=torch.float32, device=device), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn((1, T, H, D), dtype=torch.float32, device=device), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn((1, T, H, D), dtype=torch.bfloat16, device=device)
    g = torch.randn((1, T, H, D), dtype=torch.bfloat16, device=device)
    beta = torch.randn((1, T, H), dtype=torch.bfloat16, device=device)
    A_log = torch.rand(H, dtype=torch.float32, device=device)
    dt_bias = torch.rand(H, D, dtype=torch.float32, device=device)
    scale = 1.0 / math.sqrt(D)
    return q, k, v, g, beta, A_log, dt_bias, scale


def _make_state(shape, dtype):
    """Create a state tensor with the given shape and dtype."""
    n_elems = 1
    for s in shape:
        n_elems *= s
    return torch.arange(n_elems, dtype=torch.float32, device="cuda").reshape(shape).to(torch.bfloat16).to(dtype)


# state_in x state_out combinations: (has_in, has_out)
STATE_IO = [
    (True, True),
    (True, False),
    (False, True),
    (False, False),
]
STATE_IO_IDS = ["in+out", "in_only", "out_only", "no_state"]

STATE_DTYPES = ["bf16", "fp32"]

H_VALUES = [1, 4, 32, 96]
T_VALUES = [16, 64, 256, 1024, 4096, 8192, 17, 37, 97]


# ---------------------------------------------------------------------------
# Fixed-length tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("H", H_VALUES, ids=[f"H{h}" for h in H_VALUES])
@pytest.mark.parametrize("T", T_VALUES, ids=[f"T{t}" for t in T_VALUES])
@pytest.mark.parametrize("state_dtype", STATE_DTYPES)
@pytest.mark.parametrize("has_in,has_out", STATE_IO, ids=STATE_IO_IDS)
def test_fwd_fixed(T, H, state_dtype, has_in, has_out):
    q, k, v, g, beta, A_log, dt_bias, scale = _make_inputs(T, H)
    dtype = torch.bfloat16 if state_dtype == "bf16" else torch.float32

    init_k = _make_state((1, H, D, D), dtype).clone() if has_in else None
    init_r = _make_state((1, H, D, D), dtype).clone() if has_in else None
    final_k = torch.zeros(1, H, D, D, dtype=dtype, device="cuda") if has_out else None
    final_r = torch.zeros(1, H, D, D, dtype=dtype, device="cuda") if has_out else None

    out_kernel = torch.zeros_like(q)
    flash_kda.fwd(q, k, v, g, beta, scale, out_kernel,
                  A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                  initial_state=init_k, final_state=final_k)
    torch.cuda.synchronize()

    out_ref = torch.zeros_like(q)
    torch_ref(q, k, v, g, beta, scale, out_ref,
              A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
              initial_state=init_r, final_state=final_r)

    assert torch.equal(out_kernel, out_ref), \
        f"output mismatch: T={T} H={H} dtype={state_dtype} in={has_in} out={has_out}"
    if final_k is not None:
        assert torch.equal(final_k, final_r), \
            f"final_state mismatch: T={T} H={H} dtype={state_dtype} in={has_in} out={has_out}"


# ---------------------------------------------------------------------------
# Variable-length tests
# ---------------------------------------------------------------------------

VARLEN_CASES = [
    [16],
    [16, 16],
    [4, 8, 12],
    [64, 128, 256],
    [1300, 547, 2048, 963, 271, 3063],
    [1024] * 8,
    [20, 50, 100],
    [17, 33, 65],
]
VARLEN_H = [1, 4, 96]


@pytest.mark.parametrize("H", VARLEN_H, ids=[f"H{h}" for h in VARLEN_H])
@pytest.mark.parametrize("seq_lens", VARLEN_CASES,
                         ids=[f"seqs{'_'.join(str(s) for s in sl)}" for sl in VARLEN_CASES])
@pytest.mark.parametrize("state_dtype", STATE_DTYPES)
@pytest.mark.parametrize("has_in,has_out", STATE_IO, ids=STATE_IO_IDS)
def test_fwd_varlen(seq_lens, H, state_dtype, has_in, has_out):
    T_total = sum(seq_lens)
    N = len(seq_lens)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
        dtype=torch.long, device="cuda",
    )

    q, k, v, g, beta, A_log, dt_bias, scale = _make_inputs(T_total, H)
    dtype = torch.bfloat16 if state_dtype == "bf16" else torch.float32

    init_k = _make_state((N, H, D, D), dtype).clone() if has_in else None
    init_r = _make_state((N, H, D, D), dtype).clone() if has_in else None
    final_k = torch.zeros(N, H, D, D, dtype=dtype, device="cuda") if has_out else None
    final_r = torch.zeros(N, H, D, D, dtype=dtype, device="cuda") if has_out else None

    out_kernel = torch.zeros_like(q)
    flash_kda.fwd(q, k, v, g, beta, scale, out_kernel,
                  A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                  initial_state=init_k, final_state=final_k, cu_seqlens=cu_seqlens)
    torch.cuda.synchronize()

    out_ref = torch.zeros_like(q)
    torch_ref(q, k, v, g, beta, scale, out_ref,
              A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
              initial_state=init_r, final_state=final_r, cu_seqlens=cu_seqlens)

    assert torch.equal(out_kernel, out_ref), \
        f"output mismatch: seqs={seq_lens} H={H} dtype={state_dtype} in={has_in} out={has_out}"
    if final_k is not None:
        assert torch.equal(final_k, final_r), \
            f"final_state mismatch: seqs={seq_lens} H={H} dtype={state_dtype} in={has_in} out={has_out}"


# ---------------------------------------------------------------------------
# Batched tests (B > 1, equal-length sequences)
# ---------------------------------------------------------------------------

BATCH_CASES = [
    (2, 64),
    (2, 1024),
    (4, 256),
    (8, 128),
]
BATCH_H = [1, 4, 96]


@pytest.mark.parametrize("H", BATCH_H, ids=[f"H{h}" for h in BATCH_H])
@pytest.mark.parametrize("B,T", BATCH_CASES, ids=[f"B{b}_T{t}" for b, t in BATCH_CASES])
@pytest.mark.parametrize("state_dtype", STATE_DTYPES)
@pytest.mark.parametrize("has_in,has_out", STATE_IO, ids=STATE_IO_IDS)
def test_fwd_batched(B, T, H, state_dtype, has_in, has_out):
    torch.manual_seed(42)
    dtype = torch.bfloat16 if state_dtype == "bf16" else torch.float32

    q = F.normalize(torch.randn((B, T, H, D), dtype=torch.float32, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn((B, T, H, D), dtype=torch.float32, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn((B, T, H, D), dtype=torch.bfloat16, device="cuda")
    g = torch.randn((B, T, H, D), dtype=torch.bfloat16, device="cuda")
    beta = torch.randn((B, T, H), dtype=torch.bfloat16, device="cuda")
    A_log = torch.rand(H, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(H, D, dtype=torch.float32, device="cuda")
    scale = 1.0 / math.sqrt(D)

    init_k = _make_state((B, H, D, D), dtype).clone() if has_in else None
    init_r = _make_state((B, H, D, D), dtype).clone() if has_in else None
    final_k = torch.zeros(B, H, D, D, dtype=dtype, device="cuda") if has_out else None
    final_r = torch.zeros(B, H, D, D, dtype=dtype, device="cuda") if has_out else None

    # batched flash_kda (B > 1, auto cu_seqlens)
    out_kernel = torch.zeros_like(q)
    flash_kda.fwd(q, k, v, g, beta, scale, out_kernel,
                  A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                  initial_state=init_k, final_state=final_k)
    torch.cuda.synchronize()

    # torch ref
    out_ref = torch.zeros_like(q)
    torch_ref(q, k, v, g, beta, scale, out_ref,
              A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
              initial_state=init_r, final_state=final_r)

    assert torch.equal(out_kernel, out_ref), \
        f"output mismatch: B={B} T={T} H={H} dtype={state_dtype} in={has_in} out={has_out}"
    if final_k is not None:
        assert torch.equal(final_k, final_r), \
            f"final_state mismatch: B={B} T={T} H={H} dtype={state_dtype} in={has_in} out={has_out}"


# ---------------------------------------------------------------------------
# Long-sequence tests (H=1, in+out bf16 only)
# ---------------------------------------------------------------------------

LONG_T_VALUES = [131072, 1048576]
LONG_VARLEN_CASES = [
    [131072],
    [524288, 524288],
]


@pytest.mark.parametrize("T", LONG_T_VALUES, ids=[f"T{t}" for t in LONG_T_VALUES])
def test_fwd_long(T):
    H = 1
    q, k, v, g, beta, A_log, dt_bias, scale = _make_inputs(T, H)

    init_k = _make_state((1, H, D, D), torch.bfloat16).clone()
    init_r = _make_state((1, H, D, D), torch.bfloat16).clone()
    final_k = torch.zeros(1, H, D, D, dtype=torch.bfloat16, device="cuda")
    final_r = torch.zeros(1, H, D, D, dtype=torch.bfloat16, device="cuda")

    out_kernel = torch.zeros_like(q)
    flash_kda.fwd(q, k, v, g, beta, scale, out_kernel,
                  A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                  initial_state=init_k, final_state=final_k)
    torch.cuda.synchronize()

    out_ref = torch.zeros_like(q)
    torch_ref(q, k, v, g, beta, scale, out_ref,
              A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
              initial_state=init_r, final_state=final_r)

    assert torch.equal(out_kernel, out_ref), f"output mismatch: T={T}"
    assert torch.equal(final_k, final_r), f"final_state mismatch: T={T}"


@pytest.mark.parametrize("seq_lens", LONG_VARLEN_CASES,
                         ids=[f"seqs{'_'.join(str(s) for s in sl)}" for sl in LONG_VARLEN_CASES])
def test_fwd_long_varlen(seq_lens):
    H = 1
    T_total = sum(seq_lens)
    N = len(seq_lens)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
        dtype=torch.long, device="cuda",
    )

    q, k, v, g, beta, A_log, dt_bias, scale = _make_inputs(T_total, H)

    init_k = _make_state((N, H, D, D), torch.bfloat16).clone()
    init_r = _make_state((N, H, D, D), torch.bfloat16).clone()
    final_k = torch.zeros(N, H, D, D, dtype=torch.bfloat16, device="cuda")
    final_r = torch.zeros(N, H, D, D, dtype=torch.bfloat16, device="cuda")

    out_kernel = torch.zeros_like(q)
    flash_kda.fwd(q, k, v, g, beta, scale, out_kernel,
                  A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                  initial_state=init_k, final_state=final_k, cu_seqlens=cu_seqlens)
    torch.cuda.synchronize()

    out_ref = torch.zeros_like(q)
    torch_ref(q, k, v, g, beta, scale, out_ref,
              A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
              initial_state=init_r, final_state=final_r, cu_seqlens=cu_seqlens)

    assert torch.equal(out_kernel, out_ref), f"output mismatch: seqs={seq_lens}"
    assert torch.equal(final_k, final_r), f"final_state mismatch: seqs={seq_lens}"


# ---------------------------------------------------------------------------
# Row-strided Q/K/V
# ---------------------------------------------------------------------------

def _make_strided_inputs(B, T, H, layout):
    torch.manual_seed(42)
    if layout == "packed":
        packed = torch.randn(B, T, 3, H, 128, device="cuda", dtype=torch.bfloat16)
        qkv = packed.unbind(2)
    else:
        qkv = tuple(
            torch.empty(B * T * (gap * H * 128 + 8) + 8, device="cuda", dtype=torch.bfloat16)[8:]
            .as_strided((B, T, H, 128), (T * (gap * H * 128 + 8), gap * H * 128 + 8, 128, 1))
            .normal_()
            for gap in (1, 2, 3)
        )
    qkv = tuple(x.as_strided(x.shape, (
        x.stride(0) if B > 1 else 0, x.stride(1) if T > 1 else 0,
        x.stride(2) if H > 1 else 0, 1,
    )) for x in qkv)
    g = torch.randn(B, T, H, 128, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(B, T, 2, H, device="cuda", dtype=torch.bfloat16)[:, :, 0]
    A_log = torch.zeros(H, device="cuda")
    dt_bias = torch.zeros(H, 128, device="cuda")
    return qkv, (g, beta, A_log, dt_bias)


def _run_strided_fwd(qkv, gates, lengths, state, graph=False):
    q, k, v = qkv
    B, T, H, D = q.shape
    cu = None if lengths is None else torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32,
    )
    N = B if lengths is None else len(lengths)
    workspace = torch.empty(flash_kda.get_workspace_size(B * T, H, N), device="cuda", dtype=torch.uint8)
    out = torch.empty(q.shape, device="cuda", dtype=q.dtype)
    final = torch.empty_like(state)
    checkpoint = torch.zeros_like(state)
    offsets = torch.tensor(
        [min(n, 16) for n in (lengths if lengths is not None else [T] * B)],
        device="cuda", dtype=torch.int32 if cu is not None else torch.int64,
    )
    g, beta, A_log, dt_bias = gates

    def call():
        torch.ops.flash_kda.fwd(
            q, k, v, g, beta, 1 / math.sqrt(D), out, workspace, A_log, dt_bias, -5.0,
            state, final, cu, checkpoint, offsets,
        )

    if graph:
        call()
        torch.cuda.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            call()
        # These allocations precede capture and are not owned by the graph pool.
        return (out, final, checkpoint), captured, (workspace, cu, offsets)
    call()
    return out, final, checkpoint


@pytest.mark.parametrize("B,T,H,lengths", [
    (1, 37, 4, None), (2, 37, 4, None), (2, 1, 4, None),
    (1, 37, 1, None), (1, 129, 4, [17, 47, 65]), (1, 2048, 32, [2048]),
    (1, 37, 256, None), (1, 1, 1, None),
])
@pytest.mark.parametrize("layout", ["packed", "independent"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_row_strided_qkv_matches_dense(B, T, H, lengths, layout, dtype):
    qkv, gates = _make_strided_inputs(B, T, H, layout)
    N = B if lengths is None else len(lengths)
    state = torch.randn(N, H, 128, 128, device="cuda", dtype=dtype) * 0.01
    expected = _run_strided_fwd(tuple(x.contiguous() for x in qkv), gates, lengths, state)
    actual = _run_strided_fwd(qkv, gates, lengths, state)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("contiguous", [False, True])
def test_row_strided_qkv_graph_replay_uses_new_values(contiguous):
    qkv, gates = _make_strided_inputs(1, 37, 4, "packed")
    if contiguous:
        qkv = tuple(x.contiguous() for x in qkv)
    state = torch.zeros(2, 4, 128, 128, device="cuda")
    outputs, graph, _buffers = _run_strided_fwd(qkv, gates, [17, 20], state, graph=True)
    for _ in range(2):
        for x in qkv:
            x.normal_()
        graph.replay()
        expected = _run_strided_fwd(tuple(x.contiguous() for x in qkv), gates, [17, 20], state)
        for a, b in zip(outputs, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["inner", "heads", "overlap", "unaligned_row", "unaligned_base", "batch_gap"])
def test_unsupported_qkv_layout_is_rejected(layout):
    qkv, gates = _make_strided_inputs(2, 37, 4, "packed")
    if layout == "inner":
        bad = torch.empty(2, 37, 4, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
    elif layout == "heads":
        bad = torch.empty(2, 37, 8, 128, device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    elif layout == "unaligned_base":
        bad = torch.empty(2 * 37 * 4 * 128 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(2, 37, 4, 128)
    elif layout == "batch_gap":
        bad = torch.empty(2, 38, 4, 128, device="cuda", dtype=torch.bfloat16)[:, :37]
    else:
        row_stride = 256 if layout == "overlap" else 513
        bad = torch.empty_strided(
            (2, 37, 4, 128), (37 * row_stride, row_stride, 128, 1),
            device="cuda", dtype=torch.bfloat16,
        )
    state = torch.zeros(2, 4, 128, 128, device="cuda")
    with pytest.raises(RuntimeError, match="Q/K/V"):
        _run_strided_fwd((bad, *qkv[1:]), gates, None, state)
