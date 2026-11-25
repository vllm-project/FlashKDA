#include <type_traits>

#include "fwd.h"
#include "fwd_kernel1.cuh"
#include "fwd_kernel2.cuh"

// Shared body of launch_fwd, launch_prepare and launch_recurrence. RunPrepare
// and RunRecurrence select the kernels; a kernel that does not run never
// touches its inputs, so they may be null.
template <
    bool RunPrepare,
    bool RunRecurrence,
    int D,
    bool StateFP32,
    bool HasCheckpoint>
static void launch_kernels(
    cutlass::bfloat16_t const* q_ptr,
    cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    void const* initial_state_ptr,
    float scale,
    void* final_state_ptr,
    void* checkpoint_state_ptr,
    int32_t const* checkpoint_offsets_ptr,
    cutlass::bfloat16_t* out_ptr,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    int64_t q_token_stride,
    int64_t k_token_stride,
    int64_t v_token_stride,
    int32_t const* cu_seqlens_ptr,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    int num_sms,
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
    auto input_layout = [&](int64_t token_stride) {
        return make_layout(make_shape(H, T_total, D),
                           make_stride(D, token_stride ? token_stride : int64_t(D) * H, 1));
    };
    // 1D beta layout: [H*T] contiguous
    auto beta_gmem_layout = make_layout(make_shape(H * T_total));
    auto state_gmem_layout = make_layout(make_shape(N * H, D, D), LayoutRight{});

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

    const bool has_state_in = initial_state_ptr != nullptr;
    const bool has_state_out = final_state_ptr != nullptr;

    // --- State TMA descriptors (dummy pointer when state is absent)
    auto make_state_tma = [&](auto state_smem_layout,
                              auto fp32_state_smem_layout) {
        if constexpr (StateFP32) {
            auto state_ptr_load = has_state_in
                ? static_cast<float const*>(initial_state_ptr)
                : reinterpret_cast<float const*>(out_ptr);
            auto state_ptr_store = has_state_out
                ? static_cast<float*>(final_state_ptr)
                : reinterpret_cast<float*>(out_ptr);
            auto m_initial_fp32 = make_tensor(
                make_gmem_ptr(state_ptr_load), state_gmem_layout);
            auto m_final_fp32 = make_tensor(
                make_gmem_ptr(state_ptr_store), state_gmem_layout);
            auto tma_load = make_tma_copy(
                SM90_TMA_LOAD{}, m_initial_fp32, fp32_state_smem_layout);
            auto tma_store = make_tma_copy(
                SM90_TMA_STORE{}, m_final_fp32, fp32_state_smem_layout);
            return cute::make_tuple(tma_load, tma_store);
        } else {
            auto state_ptr_load = has_state_in
                ? static_cast<BF16 const*>(initial_state_ptr)
                : reinterpret_cast<BF16 const*>(out_ptr);
            auto state_ptr_store = has_state_out
                ? static_cast<BF16*>(final_state_ptr)
                : reinterpret_cast<BF16*>(out_ptr);
            auto m_init = make_tensor(make_gmem_ptr(state_ptr_load), state_gmem_layout);
            auto m_final = make_tensor(make_gmem_ptr(state_ptr_store), state_gmem_layout);
            auto tma_load = make_tma_copy(
                SM90_TMA_LOAD{}, m_init, state_smem_layout);
            auto tma_store = make_tma_copy(
                SM90_TMA_STORE{}, m_final, state_smem_layout);
            return cute::make_tuple(tma_load, tma_store);
        }
    };
    // ===== Launch Kernel 1 (prepare) =====
#if BLOCK_LEVEL_K1 >= 0
    if constexpr (RunPrepare) {
        Tensor m_q = make_tensor(make_gmem_ptr(q_ptr), input_layout(q_token_stride));
        Tensor m_k = make_tensor(make_gmem_ptr(k_ptr), input_layout(k_token_stride));
        auto tma_load_q    = make_tma_copy(SM90_TMA_LOAD{}, m_q, TMAQKLayout{});
        auto tma_load_k    = make_tma_copy(SM90_TMA_LOAD{}, m_k, TMAQKLayout{});
        auto tma_load_beta = make_tma_copy(SM90_TMA_LOAD{}, m_beta, TMABetaSmemLayout{});

        Tensor m_g = make_tensor(make_gmem_ptr(g_bf16_ptr), gmem_layout);
        auto tma_load_g = make_tma_copy(SM90_TMA_LOAD{}, m_g, TMAQKLayout{});

        auto dt_bias_gmem_layout = make_layout(make_shape(H, D), LayoutRight{});
        Tensor m_dt_bias = make_tensor(make_gmem_ptr(dt_bias_ptr), dt_bias_gmem_layout);
        auto tma_load_dt_bias = make_tma_copy(SM90_TMA_LOAD{}, m_dt_bias, TMAGTotalSmemLayout{});

        constexpr int kK1Threads = 128;
        using SharedStorageK1T = SharedStorageK1<K1L>;
        int smem_size_k1 = sizeof(SharedStorageK1T);

        auto kernel1 = _flash_kda_fwd_prepare<
            decltype(tma_load_q), decltype(tma_load_k),
            decltype(tma_load_beta),
            decltype(tma_load_g), decltype(tma_load_dt_bias),
            CHUNK, D, kK1Threads
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
    if constexpr (RunRecurrence) {
        // TMA descriptors for Kernel 2 inputs and outputs. Workspace payloads
        // use the raw bulk-copy pointers above rather than tensor maps.
        Tensor m_v   = make_tensor(make_gmem_ptr(v_ptr), input_layout(v_token_stride));
        Tensor m_out = make_tensor(make_gmem_ptr(out_ptr), gmem_layout);
        auto tma_load_v     = make_tma_copy(SM90_TMA_LOAD{}, m_v, TMAVOLayout{});
        auto tma_load_beta2 = make_tma_copy(SM90_TMA_LOAD{}, m_beta, TMABetaSmemLayout{});
        auto tma_store_out = make_tma_copy(SM90_TMA_STORE{}, m_out, TMAVOLayout{});
        auto [tma_load_initial_state, tma_store_final_state] = make_state_tma(
            TMAStateSmemLayout{}, TMAFP32StateSmemLayout{});

        // Full width: one math warpgroup plus one load/store warpgroup, for
        // register reallocation. V-split: math warpgroup plus load/store warps.
        constexpr int kK2Threads = 128 + 128;
        constexpr int kK2VSplitThreads = 32 * 2 + 128;
        dim3 block_k2(kK2Threads);

        // V-split doubles the blocks per head. Use it while its grid fits in
        // one wave; check occupancy before constructing V-split TensorMaps.
        const int vsplit_blocks = H * (D / VD) * N;
        if (vsplit_blocks <= 2 * num_sms) {
            using TmaLoadVVSplit = decltype(make_tma_copy(
                SM90_TMA_LOAD{}, m_v, K2VSplitTMAVOLayout{}));
            using TmaStoreOutVSplit = decltype(make_tma_copy(
                SM90_TMA_STORE{}, m_out, K2VSplitTMAVOLayout{}));
            using StateTmaVSplit = decltype(make_state_tma(
                K2VSplitTMAStateSmemLayout{}, K2VSplitTMAFP32StateSmemLayout{}));
            using TmaLoadStateVSplit = std::tuple_element_t<0, StateTmaVSplit>;
            using TmaStoreStateVSplit = std::tuple_element_t<1, StateTmaVSplit>;

            using SharedStorageK2T = SharedStorageK2<
                K2VSplitL, kInputStages, kOutputStages>;
            int smem_size_k2 = sizeof(SharedStorageK2T);

            auto kernel2 = _flash_kda_fwd_recurrence<
                TmaLoadVVSplit,
                decltype(tma_load_beta2),
                TmaLoadStateVSplit,
                TmaStoreStateVSplit,
                TmaStoreOutVSplit,
                CHUNK, D, kInputStages, kOutputStages, kK2VSplitThreads,
                StateFP32, HasCheckpoint, VD>;

            int blocks_per_sm = 1;
            if (vsplit_blocks > num_sms) {
                cudaFuncSetAttribute(
                    kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem_size_k2);
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks_per_sm, kernel2, kK2VSplitThreads, smem_size_k2);
            }
            if (vsplit_blocks <= blocks_per_sm * num_sms) {
                auto tma_load_v_vsplit = make_tma_copy(
                    SM90_TMA_LOAD{}, m_v, K2VSplitTMAVOLayout{});
                auto tma_store_out_vsplit = make_tma_copy(
                    SM90_TMA_STORE{}, m_out, K2VSplitTMAVOLayout{});
                auto [tma_load_initial_state_vsplit,
                      tma_store_final_state_vsplit] =
                    make_state_tma(K2VSplitTMAStateSmemLayout{},
                                   K2VSplitTMAFP32StateSmemLayout{});

                cudaFuncSetAttribute(
                    kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem_size_k2);
                dim3 grid_k2(H * (D / VD), N);
                kernel2<<<grid_k2, dim3(kK2VSplitThreads), smem_size_k2, stream>>>(
                    tma_load_v_vsplit, tma_load_beta2,
                    tma_load_initial_state_vsplit,
                    tma_store_final_state_vsplit,
                    tma_store_out_vsplit,
                    out_ptr, checkpoint_state_ptr, checkpoint_offsets_ptr,
                    T_total, H, N, cu_seqlens_ptr, total_tiles,
                    has_state_in, has_state_out,
                    ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk);
                return;
            }
        }

        using SharedStorageK2T = SharedStorageK2<K2L, kInputStages, kOutputStages>;
        int smem_size_k2 = sizeof(SharedStorageK2T);

        auto kernel2 = _flash_kda_fwd_recurrence<
            decltype(tma_load_v), decltype(tma_load_beta2),
            decltype(tma_load_initial_state),
            decltype(tma_store_final_state),
            decltype(tma_store_out),
            CHUNK, D, kInputStages, kOutputStages, kK2Threads,
            StateFP32, HasCheckpoint
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
            T_total, H, N, cu_seqlens_ptr, total_tiles,
            has_state_in, has_state_out,
            ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk
        );
    }
#endif
}

// ==================== launch_fwd ====================
template <
    int D,
    StateDtype StateT,
    bool HasCheckpoint>
void launch_fwd(
    cutlass::bfloat16_t const* q_ptr,
    cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    void const* initial_state_ptr,
    float scale,
    void* final_state_ptr,
    void* checkpoint_state_ptr,
    int32_t const* checkpoint_offsets_ptr,
    cutlass::bfloat16_t* out_ptr,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    int64_t q_token_stride,
    int64_t k_token_stride,
    int64_t v_token_stride,
    int32_t const* cu_seqlens_ptr,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    int num_sms,
    cudaStream_t stream
) {
    launch_kernels<true, true, D, StateT == StateDtype::FP32, HasCheckpoint>(
        q_ptr, k_ptr, v_ptr, g_bf16_ptr, beta_ptr, initial_state_ptr, scale,
        final_state_ptr, checkpoint_state_ptr, checkpoint_offsets_ptr, out_ptr,
        workspace_ptr, total_tiles, T_total, H, N,
        q_token_stride, k_token_stride, v_token_stride, cu_seqlens_ptr, A_log_ptr,
        dt_bias_ptr, gate_scale, num_sms, stream);
}

// ==================== launch_prepare ====================
template <int D>
void launch_prepare(
    cutlass::bfloat16_t const* q_ptr,
    cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    float scale,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    int64_t q_token_stride,
    int64_t k_token_stride,
    int32_t const* cu_seqlens_ptr,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    cudaStream_t stream
) {
    launch_kernels<true, false, D, false, false>(
        q_ptr, k_ptr, nullptr, g_bf16_ptr, beta_ptr, nullptr, scale, nullptr,
        nullptr, nullptr, nullptr, workspace_ptr, total_tiles, T_total, H, N,
        q_token_stride, k_token_stride, 0, cu_seqlens_ptr, A_log_ptr, dt_bias_ptr,
        gate_scale, 0, stream);
}

// ==================== launch_recurrence ====================
template <
    int D,
    StateDtype StateT,
    bool HasCheckpoint>
void launch_recurrence(
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    void const* initial_state_ptr,
    void* final_state_ptr,
    void* checkpoint_state_ptr,
    int32_t const* checkpoint_offsets_ptr,
    cutlass::bfloat16_t* out_ptr,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    int64_t v_token_stride,
    int32_t const* cu_seqlens_ptr,
    int num_sms,
    cudaStream_t stream
) {
    launch_kernels<false, true, D, StateT == StateDtype::FP32, HasCheckpoint>(
        nullptr, nullptr, v_ptr, nullptr, beta_ptr, initial_state_ptr, 0.0f,
        final_state_ptr, checkpoint_state_ptr, checkpoint_offsets_ptr, out_ptr,
        workspace_ptr, total_tiles, T_total, H, N, 0, 0, v_token_stride,
        cu_seqlens_ptr, nullptr, nullptr, 0.0f, num_sms, stream);
}

// Explicit instantiations: 1 K1 + 8 K2 ({256t, 192t} x {BF16, FP32} x {CKPT=0, 1})
#define INSTANTIATE_LAUNCH_FWD(D, STATE_T, CKPT) \
    template void launch_fwd<D, STATE_T, CKPT>( \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        cutlass::bfloat16_t const*, void const*, float, void*, \
        void*, int32_t const*, cutlass::bfloat16_t*, void*, \
        int, int, int, int, int64_t, int64_t, int64_t, \
        int32_t const*, float const*, float const*, float, int, \
        cudaStream_t);

#define INSTANTIATE_LAUNCH_RECURRENCE(D, STATE_T, CKPT) \
    template void launch_recurrence<D, STATE_T, CKPT>( \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        void const*, void*, void*, int32_t const*, \
        cutlass::bfloat16_t*, void*, int, int, int, int, \
        int64_t, int32_t const*, int, cudaStream_t);

#define INSTANTIATE_VARIANTS(STATE_T, CKPT) \
    INSTANTIATE_LAUNCH_FWD(128, STATE_T, CKPT) \
    INSTANTIATE_LAUNCH_RECURRENCE(128, STATE_T, CKPT)

INSTANTIATE_VARIANTS(StateDtype::BF16, false)
INSTANTIATE_VARIANTS(StateDtype::BF16, true)
INSTANTIATE_VARIANTS(StateDtype::FP32, false)
INSTANTIATE_VARIANTS(StateDtype::FP32, true)

template void launch_prepare<128>(
    cutlass::bfloat16_t const*, cutlass::bfloat16_t const*,
    cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, float,
    void*, int, int, int, int, int64_t, int64_t,
    int32_t const*, float const*, float const*, float, cudaStream_t);
