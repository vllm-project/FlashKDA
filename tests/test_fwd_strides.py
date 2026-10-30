"""Row-strided inputs must match dense inputs without changing the arithmetic."""

import math

import pytest
import torch

import flash_kda


def _make_inputs(B, T, H, layout):
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


def _run(qkv, gates, lengths, state, graph=False):
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
    qkv, gates = _make_inputs(B, T, H, layout)
    N = B if lengths is None else len(lengths)
    state = torch.randn(N, H, 128, 128, device="cuda", dtype=dtype) * 0.01
    expected = _run(tuple(x.contiguous() for x in qkv), gates, lengths, state)
    actual = _run(qkv, gates, lengths, state)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("contiguous", [False, True])
def test_row_strided_qkv_graph_replay_uses_new_values(contiguous):
    qkv, gates = _make_inputs(1, 37, 4, "packed")
    if contiguous:
        qkv = tuple(x.contiguous() for x in qkv)
    state = torch.zeros(2, 4, 128, 128, device="cuda")
    outputs, graph, _buffers = _run(qkv, gates, [17, 20], state, graph=True)
    for _ in range(2):
        for x in qkv:
            x.normal_()
        graph.replay()
        expected = _run(tuple(x.contiguous() for x in qkv), gates, [17, 20], state)
        for a, b in zip(outputs, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["inner", "heads", "overlap", "unaligned_row", "unaligned_base", "batch_gap"])
def test_unsupported_qkv_layout_is_rejected(layout):
    qkv, gates = _make_inputs(2, 37, 4, "packed")
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
    with pytest.raises(RuntimeError, match="q .*stride|q .*align|q .*row"):
        _run((bad, *qkv[1:]), gates, None, state)
