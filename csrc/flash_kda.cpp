#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor_inl.h>

#include "flash_kda.h"
#include "fwd.h"

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

int64_t get_workspace_size(
    int64_t T_total,
    int64_t H,
    int64_t N
) {
    constexpr int CHUNK = 16;
    constexpr int D = 128;

    // Upper bound: each of N sequences adds at most 1 extra tile vs floor division
    int64_t total_tiles = (T_total + CHUNK - 1) / CHUNK + N;

    static_assert(CHUNK * D * 2 % 128 == 0, "k_decayed/q_decayed/k_restored size must be 128-byte aligned");
    static_assert(D * 4 % 128 == 0, "g_total size must be 128-byte aligned");
    static_assert(CHUNK * CHUNK * 2 % 128 == 0, "INV/Mqk size must be 128-byte aligned");

    int64_t per_tile_bytes = 3 * CHUNK * D * 2 + D * 4 + 2 * CHUNK * CHUNK * 2;

    return H * total_tiles * per_tile_bytes;
}

void fwd(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    double scale,
    const Tensor& out,
    const Tensor& workspace,
    const Tensor& A_log,
    const Tensor& dt_bias,
    double lower_bound,
    std::optional<Tensor> initial_state,
    std::optional<Tensor> final_state,
    std::optional<Tensor> cu_seqlens,
    std::optional<Tensor> checkpoint_state,
    std::optional<Tensor> checkpoint_offsets
) {
    STD_TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && g.is_cuda() && beta.is_cuda() && out.is_cuda() && workspace.is_cuda(),
                    "all tensors must be on CUDA");
    STD_TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && g.is_contiguous() && out.is_contiguous() && workspace.is_contiguous(),
                    "q, k, v, g, out, and workspace must be contiguous");
    STD_TORCH_CHECK(q.scalar_type() == ScalarType::BFloat16, "q must be bfloat16");
    STD_TORCH_CHECK(k.scalar_type() == ScalarType::BFloat16, "k must be bfloat16");
    STD_TORCH_CHECK(v.scalar_type() == ScalarType::BFloat16, "v must be bfloat16");
    STD_TORCH_CHECK(g.scalar_type() == ScalarType::BFloat16, "g must be bfloat16");
    STD_TORCH_CHECK(beta.scalar_type() == ScalarType::BFloat16, "beta must be bfloat16");
    STD_TORCH_CHECK(out.scalar_type() == ScalarType::BFloat16, "out must be bfloat16");

    // Validate state tensors if present
    bool has_state_in = initial_state.has_value();
    bool has_state_out = final_state.has_value();
    bool has_checkpoint = checkpoint_state.has_value();
    STD_TORCH_CHECK(
        has_checkpoint == checkpoint_offsets.has_value(),
        "checkpoint_state and checkpoint_offsets must be provided together");
    bool state_fp32 = false;

    if (has_state_in) {
        auto& is = initial_state.value();
        STD_TORCH_CHECK(is.is_cuda() && is.is_contiguous(), "initial_state must be contiguous CUDA tensor");
        STD_TORCH_CHECK(is.scalar_type() == ScalarType::BFloat16 || is.scalar_type() == ScalarType::Float,
                        "initial_state must be bfloat16 or float32");
        if (is.scalar_type() == ScalarType::Float) state_fp32 = true;
    }
    if (has_state_out) {
        auto& fs = final_state.value();
        STD_TORCH_CHECK(fs.is_cuda() && fs.is_contiguous(), "final_state must be contiguous CUDA tensor");
        STD_TORCH_CHECK(fs.scalar_type() == ScalarType::BFloat16 || fs.scalar_type() == ScalarType::Float,
                        "final_state must be bfloat16 or float32");
        if (fs.scalar_type() == ScalarType::Float) state_fp32 = true;
    }
    if (has_checkpoint) {
        auto& cs = checkpoint_state.value();
        auto& co = checkpoint_offsets.value();
        STD_TORCH_CHECK(
            cs.is_cuda() && cs.is_contiguous(),
            "checkpoint_state must be a contiguous CUDA tensor");
        STD_TORCH_CHECK(
            cs.scalar_type() == ScalarType::Float,
            "checkpoint_state must be float32");
        STD_TORCH_CHECK(
            co.is_cuda() && co.is_contiguous(),
            "checkpoint_offsets must be a contiguous CUDA tensor");
        STD_TORCH_CHECK(
            co.scalar_type() == ScalarType::Int ||
                co.scalar_type() == ScalarType::Long,
            "checkpoint_offsets must be int32 or int64");
    }
    // If both present, dtypes must match
    if (has_state_in && has_state_out) {
        STD_TORCH_CHECK(initial_state->scalar_type() == final_state->scalar_type(),
                        "initial_state and final_state must have the same dtype");
    }

    STD_TORCH_CHECK(A_log.is_cuda() && A_log.is_contiguous(), "A_log must be contiguous CUDA tensor");
    STD_TORCH_CHECK(A_log.scalar_type() == ScalarType::Float, "A_log must be float32");
    STD_TORCH_CHECK(dt_bias.is_cuda() && dt_bias.is_contiguous(), "dt_bias must be contiguous CUDA tensor");
    STD_TORCH_CHECK(dt_bias.scalar_type() == ScalarType::Float, "dt_bias must be float32");

    // Accept 4D input [B, T, H, D]
    STD_TORCH_CHECK(q.dim() == 4, "q must be [B, T, H, D]");
    STD_TORCH_CHECK(k.dim() == 4, "k must be [B, T, H, D]");
    STD_TORCH_CHECK(v.dim() == 4, "v must be [B, T, H, D]");
    STD_TORCH_CHECK(g.dim() == 4, "g must be [B, T, H, D]");
    STD_TORCH_CHECK(beta.dim() == 3, "beta must be [B, T, H]");
    STD_TORCH_CHECK(out.dim() == 4, "out must be [B, T, H, D]");

    int64_t B = q.size(0);
    int64_t T_seq = q.size(1);
    int64_t H = q.size(2);
    int64_t D = q.size(3);
    int64_t T_total = B * T_seq;

    STD_TORCH_CHECK(k.size(0) == B && k.size(1) == T_seq && k.size(2) == H && k.size(3) == D, "k must match q shape");
    STD_TORCH_CHECK(v.size(0) == B && v.size(1) == T_seq && v.size(2) == H && v.size(3) == D, "v must match q shape");
    STD_TORCH_CHECK(g.size(0) == B && g.size(1) == T_seq && g.size(2) == H && g.size(3) == D, "g must match q shape");
    STD_TORCH_CHECK(out.size(0) == B && out.size(1) == T_seq && out.size(2) == H && out.size(3) == D, "out must match q shape");
    STD_TORCH_CHECK(beta.size(0) == B && beta.size(1) == T_seq && beta.size(2) == H,
                    "beta must be [B, T, H] matching q");

    STD_TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == H, "A_log must be [H]");
    STD_TORCH_CHECK(dt_bias.dim() == 2 && dt_bias.size(0) == H && dt_bias.size(1) == D, "dt_bias must be [H, D]");

    STD_TORCH_CHECK(D == 128, "currently only supports D == 128");

    auto q_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(q.const_data_ptr());
    auto k_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(k.const_data_ptr());
    auto v_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(v.const_data_ptr());
    auto g_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(g.const_data_ptr());
    float scale_f = scale;
    auto out_ptr = reinterpret_cast<cutlass::bfloat16_t*>(out.mutable_data_ptr());
    auto A_log_ptr = reinterpret_cast<float const*>(A_log.const_data_ptr());
    auto dt_bias_ptr = reinterpret_cast<float const*>(dt_bias.const_data_ptr());
    float gate_scale = float(lower_bound * 1.4426950408889634);

    // Transpose beta: [B, T, H] -> [H, B*T] in one materialization.
    Tensor beta_bht = torch::stable::transpose(beta, 1, 2);
    Tensor beta_hbt = torch::stable::transpose(beta_bht, 0, 1);
    Tensor beta_t = torch::stable::contiguous(beta_hbt);
    auto beta_t_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(beta_t.const_data_ptr());

    auto workspace_ptr = workspace.mutable_data_ptr();

    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(q.get_device_index(), &stream_ptr));
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    constexpr int CHUNK = 16;

    // Get state pointers (nullptr if not present)
    void const* initial_state_raw = has_state_in ? initial_state->const_data_ptr() : nullptr;
    void* final_state_raw = has_state_out ? final_state->mutable_data_ptr() : nullptr;
    float* checkpoint_state_raw = has_checkpoint
        ? static_cast<float*>(checkpoint_state->mutable_data_ptr())
        : nullptr;

    // Determine cu_seqlens and N
    bool is_varlen = cu_seqlens.has_value();
    int64_t N_val;
    void const* cu_seqlens_dev = nullptr;
    bool cu_seqlens_is_int32 = false;

    if (is_varlen) {
        STD_TORCH_CHECK(B == 1, "B must be 1 when cu_seqlens is provided");
        auto& cu_seqlens_t = cu_seqlens.value();
        STD_TORCH_CHECK(cu_seqlens_t.is_cuda(), "cu_seqlens must be on CUDA");
        STD_TORCH_CHECK(
            cu_seqlens_t.scalar_type() == ScalarType::Int ||
                cu_seqlens_t.scalar_type() == ScalarType::Long,
            "cu_seqlens must be int32 or int64");
        STD_TORCH_CHECK(cu_seqlens_t.is_contiguous(), "cu_seqlens must be contiguous");
        STD_TORCH_CHECK(cu_seqlens_t.dim() == 1, "cu_seqlens must be 1D");
        N_val = cu_seqlens_t.numel() - 1;
        STD_TORCH_CHECK(N_val > 0, "cu_seqlens must have at least 2 elements");
        cu_seqlens_is_int32 = cu_seqlens_t.scalar_type() == ScalarType::Int;
        cu_seqlens_dev = cu_seqlens_t.const_data_ptr();
    } else {
        N_val = B;
    }

    // Validate state shapes: always [N, H, D, D]
    if (has_state_in) {
        auto& is = initial_state.value();
        STD_TORCH_CHECK(is.dim() == 4, "initial_state must be [N, H, D, D]");
        STD_TORCH_CHECK(is.size(0) == N_val && is.size(1) == H && is.size(2) == D && is.size(3) == D,
                        "initial_state must be [N, H, D, D]");
    }
    if (has_state_out) {
        auto& fs = final_state.value();
        STD_TORCH_CHECK(fs.dim() == 4, "final_state must be [N, H, D, D]");
        STD_TORCH_CHECK(fs.size(0) == N_val && fs.size(1) == H && fs.size(2) == D && fs.size(3) == D,
                        "final_state must be [N, H, D, D]");
    }
    if (has_checkpoint) {
        auto& cs = checkpoint_state.value();
        auto& co = checkpoint_offsets.value();
        STD_TORCH_CHECK(
            cs.dim() == 4 && cs.size(0) == N_val && cs.size(1) == H &&
                cs.size(2) == D && cs.size(3) == D,
            "checkpoint_state must be [N, H, D, D]");
        STD_TORCH_CHECK(
            co.dim() == 1 && co.size(0) == N_val,
            "checkpoint_offsets must be [N]");
        STD_TORCH_CHECK(
            co.scalar_type() ==
                (cu_seqlens_is_int32 ? ScalarType::Int : ScalarType::Long),
            "checkpoint_offsets and cu_seqlens must have matching dtypes");
    }

    int total_tiles;
    if (is_varlen) {
        total_tiles = int((T_total + CHUNK - 1) / CHUNK + N_val);  // upper bound for varlen
    } else {
        total_tiles = int(N_val * ((T_seq + CHUNK - 1) / CHUNK));   // exact for batched
    }

    auto dispatch = [&](auto typed_cu_seqlens) {
        using SeqlenPtr = decltype(typed_cu_seqlens);
        SeqlenPtr typed_checkpoint_offsets = has_checkpoint
            ? static_cast<SeqlenPtr>(checkpoint_offsets->const_data_ptr())
            : nullptr;
        #define LAUNCH(HI, HO, FP32, VL) \
            launch_fwd<128, HI, HO, FP32, VL>( \
                q_ptr, k_ptr, v_ptr, g_ptr, beta_t_ptr, \
                initial_state_raw, scale_f, final_state_raw, \
                checkpoint_state_raw, typed_checkpoint_offsets, out_ptr, \
                workspace_ptr, total_tiles, \
                int(T_total), int(H), int(N_val), typed_cu_seqlens, \
                A_log_ptr, dt_bias_ptr, gate_scale, stream)

        #define DISPATCH_STATE(VL) \
            if (!has_state_in && !has_state_out) { \
                LAUNCH(false, false, false, VL); \
            } else if (has_state_in && has_state_out && state_fp32) { \
                LAUNCH(true, true, true, VL); \
            } else if (has_state_in && has_state_out && !state_fp32) { \
                LAUNCH(true, true, false, VL); \
            } else if (!has_state_in && has_state_out && state_fp32) { \
                LAUNCH(false, true, true, VL); \
            } else if (!has_state_in && has_state_out && !state_fp32) { \
                LAUNCH(false, true, false, VL); \
            } else if (has_state_in && !has_state_out && state_fp32) { \
                LAUNCH(true, false, true, VL); \
            } else { \
                LAUNCH(true, false, false, VL); \
            }

        if (is_varlen) {
            DISPATCH_STATE(true);
        } else {
            DISPATCH_STATE(false);
        }

        #undef DISPATCH_STATE
        #undef LAUNCH
    };

    if (cu_seqlens_is_int32) {
        dispatch(static_cast<int32_t const*>(cu_seqlens_dev));
    } else {
        dispatch(static_cast<int64_t const*>(cu_seqlens_dev));
    }
}
