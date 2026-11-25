import torch

from . import _C  # noqa: F401


def get_workspace_size(T_total, H, N=1):
    return torch.ops.flash_kda.get_workspace_size(T_total, H, N)


def fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    out,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    workspace=None,
):
    """FlashKDA forward (Flash Kimi Delta Attention).

    Args:
        q (torch.Tensor): Query, bf16, shape ``[B, T, H, K]``.
        k (torch.Tensor): Key, bf16, shape ``[B, T, H, K]``.
        v (torch.Tensor): Value, bf16, shape ``[B, T, H, V]``.
        g (torch.Tensor): Gate before activation, bf16, shape ``[B, T, H, K]``.
        beta (torch.Tensor): Beta logits (pre-activation; sigmoid is applied
            internally), bf16, shape ``[B, T, H]``.
        scale (float): Scaling factor.
        out (torch.Tensor): Output buffer, bf16, shape ``[B, T, H, V]``. Written
            in place.
        A_log (torch.Tensor): Log-gate parameter, fp32, shape ``[H]``.
        dt_bias (torch.Tensor): Gate bias, fp32, shape ``[H, K]``.
        lower_bound (float): Gate lower bound, expected in ``[-5.0, 0]``.
        initial_state (torch.Tensor, optional): Initial recurrent state, bf16
            or fp32. Shape ``[B, H, V, K]`` for batched mode, or ``[N, H, V, K]``
            for varlen mode. ``None`` means start from zero.
        final_state (torch.Tensor, optional): Output buffer for the final
            recurrent state. Same dtype/shape rules as ``initial_state``.
        cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths, int32
            or int64, shape ``[N+1]``. When provided, ``B`` must be 1.
        workspace (torch.Tensor, optional): Reusable uint8 workspace. Allocated
            automatically when omitted.
    Notes:
        * Currently requires ``K = V = 128``.
        * Q/K/V may have a padded token stride, with contiguous H/D rows,
          16-byte-aligned row starts, and a uniform stride across B*T.
          Beta may be strided. Other input and output tensors must be contiguous.
    """
    B, T_seq, H = q.shape[:3]
    if cu_seqlens is None:
        if T_seq == 0:
            cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
        else:
            cu_seqlens = torch.arange(
                0, (B + 1) * T_seq, T_seq, dtype=torch.int32, device=q.device
            )
        if B > 1:
            D = q.shape[3]
            for name, t in (("k", k), ("v", v), ("g", g), ("out", out)):
                if t.shape != (B, T_seq, H, D):
                    raise ValueError(
                        f"{name} must have shape {(B, T_seq, H, D)}, got {tuple(t.shape)}"
                    )
            if beta.shape != (B, T_seq, H):
                raise ValueError(
                    f"beta must have shape {(B, T_seq, H)}, got {tuple(beta.shape)}"
                )
            q = q.reshape(1, B * T_seq, H, D)
            k = k.reshape(1, B * T_seq, H, D)
            v = v.reshape(1, B * T_seq, H, D)
            g = g.reshape(1, B * T_seq, H, D)
            beta = beta.reshape(1, B * T_seq, H)
            out = out.view(1, B * T_seq, H, D)
    elif cu_seqlens.dtype == torch.int64:
        cu_seqlens = cu_seqlens.to(torch.int32)
    N = cu_seqlens.numel() - 1
    if workspace is None:
        workspace = torch.empty(
            get_workspace_size(B * T_seq, H, N),
            dtype=torch.uint8,
            device=q.device,
        )

    torch.ops.flash_kda.fwd(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        out,
        workspace,
        A_log,
        dt_bias,
        lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
    )
