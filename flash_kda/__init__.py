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
    cu_seqlens,
    initial_state=None,
    final_state=None,
    workspace=None,
):
    """FlashKDA forward (Flash Kimi Delta Attention).

    Args:
        q (torch.Tensor): Query, bf16, shape ``[1, T, H, K]``.
        k (torch.Tensor): Key, bf16, shape ``[1, T, H, K]``.
        v (torch.Tensor): Value, bf16, shape ``[1, T, H, V]``.
        g (torch.Tensor): Gate before activation, bf16, shape ``[1, T, H, K]``.
        beta (torch.Tensor): Beta logits (pre-activation; sigmoid is applied
            internally), bf16, shape ``[1, T, H]``.
        scale (float): Scaling factor.
        out (torch.Tensor): Output buffer, bf16, shape ``[1, T, H, V]``. Written
            in place.
        A_log (torch.Tensor): Log-gate parameter, fp32, shape ``[H]``.
        dt_bias (torch.Tensor): Gate bias, fp32, shape ``[H, K]``.
        lower_bound (float): Gate lower bound, expected in ``[-5.0, 0]``.
        cu_seqlens (torch.Tensor): Cumulative sequence lengths, int32,
            shape ``[N+1]``.
        initial_state (torch.Tensor, optional): Initial recurrent state, bf16
            or fp32, shape ``[N, H, V, K]``. ``None`` means start from zero.
        final_state (torch.Tensor, optional): Output buffer for the final
            recurrent state, shape ``[N, H, V, K]``.
        workspace (torch.Tensor, optional): Reusable uint8 workspace. Allocated
            automatically when omitted.
    Notes:
        * Currently requires ``K = V = 128`` and ``B = 1`` (packed varlen).
        * Q/K/V may have a padded token stride, with contiguous H/D rows and
          16-byte-aligned row starts. Beta may be strided. Other input and
          output tensors must be contiguous.
    """
    _, T_total, H = q.shape[:3]
    N = cu_seqlens.numel() - 1
    if workspace is None:
        workspace = torch.empty(
            get_workspace_size(T_total, H, N),
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
        cu_seqlens,
        initial_state=initial_state,
        final_state=final_state,
    )
