#include <type_traits>

#include "fwd.h"
#include "fwd_kernel1.cuh"
#include "fwd_kernel2.cuh"

// ==================== launch_fwd ====================
template <
    int D,
    bool HasStateIn,
    bool HasStateOut,
    bool StateFP32,
    bool HasCheckpoint,
    bool IsVarlen,
    typename SeqlenT>
void launch_fwd(
    cutlass::bfloat16_t const* q_ptr,
    cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    void const* initial_state_ptr,
    float scale,
    void* final_state_ptr,
    float* checkpoint_state_ptr,
    SeqlenT const* checkpoint_offsets_ptr,
    int num_checkpoints,
    cutlass::bfloat16_t* out_ptr,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    SeqlenT const* cu_seqlens_ptr,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    bool use_vsplit,
    cudaStream_t stream
) {
    using BF16 = cutlass::bfloat16_t;
    constexpr int kInputStages = 3;
    constexpr int kOutputStages = 2;
    constexpr int CHUNK = 16;
    constexpr int VD = 64;

    using K1L = K1Layouts<D, CHUNK>;
    using K2L = K2Layouts<D, CHUNK>;
    using K2VSplitL = K2Layouts<D, CHUNK, VD>;
    using WS = WorkspaceSizes<CHUNK, D>;

    // Raw bulk copies make the swizzled shared-memory layout a private ABI
    // between K1 and K2. Fail at compile time if either side changes alone.
    static_assert(std::is_same_v<typename K1L::MMALayout,
                                 typename K2L::MMALayout>);
    static_assert(std::is_same_v<typename K1L::GTotalLayout,
                                 typename K2L::GTotalLayout>);
    static_assert(std::is_same_v<typename K1L::LMLayout,
                                 typename K2L::LMLayout>);

    // TMA layouts for Kernel 1
    using TMAQKLayout = typename K1L::TMAQKLayout;
    using TMAGLayout = typename K1L::TMAGLayout;
    using TMABetaSmemLayout = typename K1L::TMABetaSmemLayout;
    using TMAGTotalSmemLayout = typename K1L::TMAGTotalSmemLayout;

    // TMA layouts for Kernel 2
    using TMAVOLayout = typename K2L::TMAVOLayout;
    using TMAStateSmemLayout = typename K2L::TMAStateSmemLayout;
    using TMAFP32StateSmemLayout = typename K2L::TMAFP32StateSmemLayout;
    using K2VSplitTMAVOLayout = typename K2VSplitL::TMAVOLayout;
    using K2VSplitTMAStateSmemLayout =
        typename K2VSplitL::TMAStateSmemLayout;
    using K2VSplitTMAFP32StateSmemLayout =
        typename K2VSplitL::TMAFP32StateSmemLayout;

    // --- gmem layouts for original tensors
    auto gmem_layout = make_layout(make_shape(H, T_total, D), make_stride(D, D * H, 1));
    // 1D beta layout: [H*T] contiguous
    auto beta_gmem_layout = make_layout(make_shape(H * T_total));
    auto state_gmem_layout = make_layout(make_shape(N * H, D, D), LayoutRight{});

    Tensor m_q   = make_tensor(make_gmem_ptr(q_ptr), gmem_layout);
    Tensor m_k   = make_tensor(make_gmem_ptr(k_ptr), gmem_layout);
    Tensor m_v   = make_tensor(make_gmem_ptr(v_ptr), gmem_layout);
    Tensor m_out = make_tensor(make_gmem_ptr(out_ptr), gmem_layout);
    Tensor m_beta = make_tensor(make_gmem_ptr<BF16>(beta_ptr), beta_gmem_layout);

    // --- Workspace gmem layouts (separated arrays)
    int64_t n_ht = int64_t(H) * total_tiles;
    char* ws = reinterpret_cast<char*>(workspace_ptr);
    BF16*  ws_kd  = reinterpret_cast<BF16*>(ws);
    BF16*  ws_qd  = reinterpret_cast<BF16*>(ws + n_ht * WS::kKDecayed);
    BF16*  ws_kr  = reinterpret_cast<BF16*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed));
    float* ws_gt  = reinterpret_cast<float*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored));
    BF16*  ws_inv = reinterpret_cast<BF16*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored + WS::kGTotal));
    BF16*  ws_mqk = reinterpret_cast<BF16*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored + WS::kGTotal + WS::kINV));

    // --- TMA descriptors for Kernel 1 inputs
    auto tma_load_q    = make_tma_copy(SM90_TMA_LOAD{}, m_q, TMAQKLayout{});
    auto tma_load_k    = make_tma_copy(SM90_TMA_LOAD{}, m_k, TMAQKLayout{});
    auto tma_load_beta = make_tma_copy(SM90_TMA_LOAD{}, m_beta, TMABetaSmemLayout{});

    Tensor m_g = make_tensor(make_gmem_ptr(g_bf16_ptr), gmem_layout);
    auto tma_load_g = make_tma_copy(SM90_TMA_LOAD{}, m_g, TMAQKLayout{});

    auto dt_bias_gmem_layout = make_layout(make_shape(H, D), LayoutRight{});
    Tensor m_dt_bias = make_tensor(make_gmem_ptr(dt_bias_ptr), dt_bias_gmem_layout);
    auto tma_load_dt_bias = make_tma_copy(SM90_TMA_LOAD{}, m_dt_bias, TMAGTotalSmemLayout{});

    // --- TMA descriptors for Kernel 2 inputs and outputs. Workspace payloads
    // use the raw bulk-copy pointers above rather than tensor maps.
    auto tma_load_v     = make_tma_copy(SM90_TMA_LOAD{}, m_v, TMAVOLayout{});
    auto tma_load_beta2 = make_tma_copy(SM90_TMA_LOAD{}, m_beta, TMABetaSmemLayout{});
    auto tma_store_out = make_tma_copy(SM90_TMA_STORE{}, m_out, TMAVOLayout{});

    // --- State TMA descriptors (conditional on HasStateIn/HasStateOut and StateFP32)
    auto make_state_tma = [&](auto state_smem_layout,
                              auto fp32_state_smem_layout) {
        if constexpr (StateFP32) {
            // FP32 state TMA descriptors
            auto m_initial_fp32 = make_tensor(
                make_gmem_ptr(static_cast<float const*>(initial_state_ptr)), state_gmem_layout);
            auto m_final_fp32 = make_tensor(
                make_gmem_ptr(static_cast<float*>(final_state_ptr)), state_gmem_layout);
            auto tma_load = make_tma_copy(
                SM90_TMA_LOAD{}, m_initial_fp32, fp32_state_smem_layout);
            auto tma_store = make_tma_copy(
                SM90_TMA_STORE{}, m_final_fp32, fp32_state_smem_layout);
            return cute::make_tuple(tma_load, tma_store);
        } else {
            // BF16 state TMA descriptors (or dummy for no-state)
            auto state_ptr_load = HasStateIn
                ? static_cast<BF16 const*>(initial_state_ptr)
                : reinterpret_cast<BF16 const*>(out_ptr);  // dummy, never used
            auto state_ptr_store = HasStateOut
                ? static_cast<BF16*>(final_state_ptr)
                : reinterpret_cast<BF16*>(out_ptr);  // dummy, never used
            auto m_init = make_tensor(make_gmem_ptr(state_ptr_load), state_gmem_layout);
            auto m_final = make_tensor(make_gmem_ptr(state_ptr_store), state_gmem_layout);
            auto tma_load = make_tma_copy(
                SM90_TMA_LOAD{}, m_init, state_smem_layout);
            auto tma_store = make_tma_copy(
                SM90_TMA_STORE{}, m_final, state_smem_layout);
            return cute::make_tuple(tma_load, tma_store);
        }
    };
    auto [tma_load_initial_state, tma_store_final_state] = make_state_tma(
        TMAStateSmemLayout{}, TMAFP32StateSmemLayout{});
    // ===== Launch Kernel 1 (prepare) =====
#if BLOCK_LEVEL_K1 >= 0
    {
        constexpr int kK1Threads = 128;
        using SharedStorageK1T = SharedStorageK1<K1L>;
        int smem_size_k1 = sizeof(SharedStorageK1T);

        auto kernel1 = _flash_kda_fwd_prepare<
            decltype(tma_load_q), decltype(tma_load_k),
            decltype(tma_load_beta),
            decltype(tma_load_g), decltype(tma_load_dt_bias),
            CHUNK, D, kK1Threads, IsVarlen, SeqlenT
        >;

        cudaFuncSetAttribute(kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_k1);

        dim3 grid_k1(total_tiles, H);
        dim3 block_k1(kK1Threads);

        kernel1<<<grid_k1, block_k1, smem_size_k1, stream>>>(
            tma_load_q, tma_load_k, tma_load_beta,
            tma_load_g, tma_load_dt_bias,
            scale, T_total, H, N, cu_seqlens_ptr, total_tiles,
            A_log_ptr, gate_scale,
            ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk
        );
    }
#endif

    // ===== Launch Kernel 2 (recurrence) =====
#if BLOCK_LEVEL_K2 >= 0
    {
        constexpr int kK2Threads = 32 * 2 + 128;
        dim3 block_k2(kK2Threads);

        if (use_vsplit) {
            // Keep the default path's host launch overhead unchanged: split
            // TensorMaps are constructed only when this path is requested.
            auto tma_load_v_vsplit = make_tma_copy(
                SM90_TMA_LOAD{}, m_v, K2VSplitTMAVOLayout{});
            auto tma_store_out_vsplit = make_tma_copy(
                SM90_TMA_STORE{}, m_out, K2VSplitTMAVOLayout{});
            auto [tma_load_initial_state_vsplit,
                  tma_store_final_state_vsplit] =
                make_state_tma(K2VSplitTMAStateSmemLayout{},
                               K2VSplitTMAFP32StateSmemLayout{});

            using SharedStorageK2T = SharedStorageK2<
                K2VSplitL, kInputStages, kOutputStages>;
            int smem_size_k2 = sizeof(SharedStorageK2T);
            dim3 grid_k2(H * (D / VD), N);

            auto kernel2 = _flash_kda_fwd_recurrence<
                decltype(tma_load_v_vsplit),
                decltype(tma_load_beta2),
                decltype(tma_load_initial_state_vsplit),
                decltype(tma_store_final_state_vsplit),
                decltype(tma_store_out_vsplit),
                CHUNK, D, kInputStages, kOutputStages, kK2Threads,
                HasStateIn, HasStateOut, StateFP32, HasCheckpoint,
                IsVarlen, SeqlenT,
                VD>;

            cudaFuncSetAttribute(
                kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size_k2);
            kernel2<<<grid_k2, block_k2, smem_size_k2, stream>>>(
                tma_load_v_vsplit, tma_load_beta2,
                tma_load_initial_state_vsplit,
                tma_store_final_state_vsplit,
                tma_store_out_vsplit,
                out_ptr, checkpoint_state_ptr, checkpoint_offsets_ptr,
                num_checkpoints,
                T_total, H, N, cu_seqlens_ptr, total_tiles,
                ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk);
            return;
        }

        using SharedStorageK2T = SharedStorageK2<K2L, kInputStages, kOutputStages>;
        int smem_size_k2 = sizeof(SharedStorageK2T);

        auto kernel2 = _flash_kda_fwd_recurrence<
            decltype(tma_load_v), decltype(tma_load_beta2),
            decltype(tma_load_initial_state),
            decltype(tma_store_final_state),
            decltype(tma_store_out),
            CHUNK, D, kInputStages, kOutputStages, kK2Threads,
            HasStateIn, HasStateOut, StateFP32, HasCheckpoint,
            IsVarlen, SeqlenT
        >;

        cudaFuncSetAttribute(kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_k2);

        // K2 maps x to head so all heads of a sequence launch together.
        // Varlen reverses y in-kernel to process vLLM's trailing prefills first.
        dim3 grid_k2(H, N);

        kernel2<<<grid_k2, block_k2, smem_size_k2, stream>>>(
            tma_load_v, tma_load_beta2,
            tma_load_initial_state,
            tma_store_final_state,
            tma_store_out,
            out_ptr, checkpoint_state_ptr, checkpoint_offsets_ptr,
            num_checkpoints,
            T_total, H, N, cu_seqlens_ptr, total_tiles,
            ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk
        );
    }
#endif
}

// Explicit instantiations
#define INSTANTIATE_LAUNCH_FWD(D, HI, HO, FP32, CKPT, VL, SEQLEN_T) \
    template void launch_fwd<D, HI, HO, FP32, CKPT, VL, SEQLEN_T>( \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        cutlass::bfloat16_t const*, void const*, float, void*, \
        float*, SEQLEN_T const*, int, cutlass::bfloat16_t*, void*, \
        int, int, int, int, \
        SEQLEN_T const*, float const*, float const*, float, bool, \
        cudaStream_t);

#define INSTANTIATE_CHECKPOINT_VARIANTS(HI, HO, FP32, VL, SEQLEN_T) \
    INSTANTIATE_LAUNCH_FWD(128, HI, HO, FP32, false, VL, SEQLEN_T) \
    INSTANTIATE_LAUNCH_FWD(128, HI, HO, FP32, true,  VL, SEQLEN_T)

#define INSTANTIATE_STATE_VARIANTS(VL, SEQLEN_T) \
    INSTANTIATE_CHECKPOINT_VARIANTS(true,  true,  false, VL, SEQLEN_T) \
    INSTANTIATE_CHECKPOINT_VARIANTS(true,  true,  true,  VL, SEQLEN_T) \
    INSTANTIATE_CHECKPOINT_VARIANTS(false, false, false, VL, SEQLEN_T) \
    INSTANTIATE_CHECKPOINT_VARIANTS(false, true,  false, VL, SEQLEN_T) \
    INSTANTIATE_CHECKPOINT_VARIANTS(true,  false, false, VL, SEQLEN_T) \
    INSTANTIATE_CHECKPOINT_VARIANTS(false, true,  true,  VL, SEQLEN_T) \
    INSTANTIATE_CHECKPOINT_VARIANTS(true,  false, true,  VL, SEQLEN_T)

INSTANTIATE_STATE_VARIANTS(true, int32_t)
INSTANTIATE_STATE_VARIANTS(true, int64_t)
INSTANTIATE_STATE_VARIANTS(false, int32_t)
INSTANTIATE_STATE_VARIANTS(false, int64_t)
