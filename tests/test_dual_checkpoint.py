import torch

import flash_kda


def _run_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    checkpoint_offsets: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch, tokens, heads, dim = q.shape
    num_sequences = batch if cu_seqlens is None else cu_seqlens.numel() - 1
    out = torch.empty_like(q)
    final_state = torch.empty_like(initial_state)
    checkpoint_state = None
    if checkpoint_offsets is not None:
        checkpoint_state = torch.empty(
            (*checkpoint_offsets.shape, heads, dim, dim),
            dtype=torch.float32,
            device=q.device,
        )
    workspace = torch.empty(
        flash_kda.get_workspace_size(batch * tokens, heads, num_sequences),
        dtype=torch.uint8,
        device=q.device,
    )
    torch.ops.flash_kda.fwd(
        q,
        k,
        v,
        g,
        beta,
        dim**-0.5,
        out,
        workspace,
        torch.zeros(heads, dtype=torch.float32, device=q.device),
        torch.zeros(heads, dim, dtype=torch.float32, device=q.device),
        -3.0,
        initial_state,
        final_state,
        cu_seqlens,
        checkpoint_state,
        checkpoint_offsets,
    )
    return final_state, checkpoint_state


@torch.inference_mode()
def test_two_checkpoints_match_prefix_final_states():
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, tokens, heads, dim = 1, 384, 8, 128
    shape = (batch, tokens, heads, dim)
    q, k, v, g = [
        torch.randn(shape, dtype=torch.bfloat16, device=device) for _ in range(4)
    ]
    beta = torch.randn(
        batch, tokens, heads, dtype=torch.bfloat16, device=device
    )
    initial_state = torch.randn(
        batch, heads, dim, dim, dtype=torch.float32, device=device
    )
    offsets = torch.tensor([[128, 256]], dtype=torch.int64, device=device)

    _, checkpoints = _run_fwd(q, k, v, g, beta, initial_state, offsets)
    assert checkpoints is not None
    for checkpoint_slot, offset in enumerate(offsets[0].tolist()):
        reference, _ = _run_fwd(
            q[:, :offset],
            k[:, :offset],
            v[:, :offset],
            g[:, :offset],
            beta[:, :offset],
            initial_state,
        )
        torch.testing.assert_close(
            checkpoints[:, checkpoint_slot], reference, rtol=0, atol=0
        )


@torch.inference_mode()
def test_two_varlen_checkpoints_match_per_sequence_prefix_states():
    """Exercise the exact packed-varlen, int32-offset path used by vLLM."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    lengths = [384, 320]
    total_tokens, heads, dim = sum(lengths), 8, 128
    shape = (1, total_tokens, heads, dim)
    q, k, v, g = [
        torch.randn(shape, dtype=torch.bfloat16, device=device) for _ in range(4)
    ]
    beta = torch.randn(
        1, total_tokens, heads, dtype=torch.bfloat16, device=device
    )
    initial_state = torch.randn(
        len(lengths), heads, dim, dim, dtype=torch.float32, device=device
    )
    cu_seqlens = torch.tensor(
        [0, lengths[0], total_tokens], dtype=torch.int32, device=device
    )
    offsets = torch.tensor(
        [[128, 256], [128, 256]], dtype=torch.int32, device=device
    )

    _, checkpoints = _run_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        offsets,
        cu_seqlens,
    )
    assert checkpoints is not None
    for sequence_idx, bos in enumerate(cu_seqlens[:-1]):
        start = int(bos.item())
        for checkpoint_slot, offset in enumerate(offsets[sequence_idx].tolist()):
            stop = start + offset
            reference, _ = _run_fwd(
                q[:, start:stop],
                k[:, start:stop],
                v[:, start:stop],
                g[:, start:stop],
                beta[:, start:stop],
                initial_state[sequence_idx : sequence_idx + 1],
            )
            torch.testing.assert_close(
                checkpoints[sequence_idx, checkpoint_slot],
                reference[0],
                rtol=0,
                atol=0,
            )
