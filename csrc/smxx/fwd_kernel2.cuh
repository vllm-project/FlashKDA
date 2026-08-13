#pragma once

#include "utils.cuh"

template <int D, int CHUNK = 16, int VD = D>
struct K2Layouts {
    using MMALayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using TransposedMMALayout = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<D>{}, Int<CHUNK>{}),
        LayoutRight{}
    ));
    using VOLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<VD>{}),
        LayoutLeft{}
    ));
    using BetaSmemLayout = Layout<Shape<Int<32>>, Stride<Int<1>>>;
    using StateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<VD>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using TransposedStateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<D>{}, Int<VD>{}),
        LayoutRight{}
    ));
    using GTotalLayout = Layout<Shape<Int<D>>, Stride<Int<1>>>;
    using LMLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<CHUNK>{}),
        LayoutLeft{}
    ));

    using TMABetaSmemLayout = BetaSmemLayout;  // 1D TMA, no dummy dim
    using TMAVOLayout = decltype(composition(
        VOLayout{}.layout_a(),
        VOLayout{}.offset(),
        prepend(VOLayout{}.layout_b())
    ));
    using TMAStateSmemLayout = decltype(composition(
        StateSmemLayout{}.layout_a(),
        StateSmemLayout{}.offset(),
        prepend(StateSmemLayout{}.layout_b())
    ));

    // FP32 state layout (K_SW32 atom, same 8x8 atom structure as K_INTER bf16)
    using FP32StateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_K_SW32_Atom<float>{},
        make_shape(Int<VD>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using TMAFP32StateSmemLayout = decltype(composition(
        FP32StateSmemLayout{}.layout_a(),
        FP32StateSmemLayout{}.offset(),
        prepend(FP32StateSmemLayout{}.layout_b())
    ));
};

template <class Layouts, int InputStages, int OutputStages>
struct SharedStorageK2 {
    using BF16 = cutlass::bfloat16_t;
    using VOLayout = typename Layouts::VOLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using MMALayout = typename Layouts::MMALayout;

    struct InputStorage {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<VOLayout>> v;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<BetaSmemLayout>> beta;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_decayed;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> q_decayed;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_restored;
        alignas(128) cute::ArrayEngine<float, cute::cosize_v<GTotalLayout>> g_total;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> INV;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> Mqk;
    };

    struct OutputStorage {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<VOLayout>> out;
    };

    // Initial-state staging is drained into resident registers before the
    // input/output pipelines reuse this storage.
    union {
        struct {
            InputStorage input[InputStages];
            OutputStorage output[OutputStages];
        };
        alignas(128) char state_staging[cute::cosize_v<StateSmemLayout> * sizeof(float)];
    };

    typename cutlass::PipelineTmaAsync<InputStages>::SharedStorage load_pipeline;
    typename cutlass::PipelineAsync<OutputStages>::SharedStorage store_pipeline;
    alignas(16) cutlass::arch::ClusterTransactionBarrier state_tma_barrier;
};

template <class CFragment, class BFragment>
CUTLASS_DEVICE void movm_transpose_c_to_b_16x16(
    CFragment const& source,
    BFragment& destination
) {
    static_assert(sizeof(CFragment) == 4 * sizeof(uint32_t));
    static_assert(sizeof(BFragment) == 4 * sizeof(uint32_t));

    auto const* source_regs = reinterpret_cast<uint32_t const*>(&source(0));
    auto* destination_regs = reinterpret_cast<uint32_t*>(&destination(0));
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        SM75_U32x1_MOVM_T::copy(source_regs[i], destination_regs[i]);
    }
}

// ==================== Kernel 2: Recurrence ====================
template <
    class TmaLoadV,
    class TmaLoadBeta,
    class TmaLoadState,
    class TmaStoreOut,
    int CHUNK,
    int D,
    int InputStages,
    int OutputStages,
    int NumThreads,
    bool HasStateIn = true,
    bool HasStateOut = true,
    bool StateFP32 = false,
    bool IsVarlen = true,
    typename SeqlenT = int64_t,
    int VD = D
>
__global__ void __launch_bounds__(NumThreads, 2) _flash_kda_fwd_recurrence(
    CUTE_GRID_CONSTANT TmaLoadV const tma_load_v,
    CUTE_GRID_CONSTANT TmaLoadBeta const tma_load_beta,
    CUTE_GRID_CONSTANT TmaLoadState const tma_load_initial_state,
    CUTE_GRID_CONSTANT TmaStoreOut const tma_store_out,
    cutlass::bfloat16_t* out_raw_ptr,
    void* final_state_raw_ptr,
    int T_total,
    int H,
    int N,
    SeqlenT const* cu_seqlens,
    int total_tiles,
    cutlass::bfloat16_t const* ws_kd,
    cutlass::bfloat16_t const* ws_qd,
    cutlass::bfloat16_t const* ws_kr,
    float const* ws_gt,
    cutlass::bfloat16_t const* ws_inv,
    cutlass::bfloat16_t const* ws_mqk
) {
    using BF16 = cutlass::bfloat16_t;
    static_assert(D % VD == 0, "VD must divide D");
    static_assert(VD == 64 || VD == D,
                  "K2 currently supports the original V tile or VD=64");
    using Layouts = K2Layouts<D, CHUNK, VD>;
    using MMALayout = typename Layouts::MMALayout;
    using TransposedMMALayout = typename Layouts::TransposedMMALayout;
    using VOLayout = typename Layouts::VOLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using TransposedStateSmemLayout = typename Layouts::TransposedStateSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using TMAVOLayout = typename Layouts::TMAVOLayout;
    using TMABetaSmemLayout = typename Layouts::TMABetaSmemLayout;
    using TMAStateSmemLayout = typename Layouts::TMAStateSmemLayout;
    using SharedStorageT = SharedStorageK2<Layouts, InputStages, OutputStages>;

    // --- shared memory
    extern __shared__ __align__(128) unsigned char shared_mem[];
    SharedStorageT& shared_storage =
        *reinterpret_cast<SharedStorageT*>(shared_mem);

    constexpr int kWarpSize = 32;
    constexpr int kComputeThreads = 128;
    constexpr int kVSlices = D / VD;
    constexpr int kValueBlocksPerWarp = VD / ((kComputeThreads / kWarpSize) * 16);
    static_assert(kValueBlocksPerWarp == 1 || kValueBlocksPerWarp == 2);

    // Transaction bytes: v + beta + k_decayed + q_decayed + k_restored + g_total + INV + Mqk
    constexpr uint32_t kTmaTransactionBytes =
        uint32_t(cute::cosize_v<VOLayout>) * uint32_t(sizeof(BF16)) +
        uint32_t(32) * uint32_t(sizeof(BF16)) +  // beta (bf16, sigmoid fused)
        uint32_t(cute::cosize_v<MMALayout>) * uint32_t(sizeof(BF16)) * 3 +
        uint32_t(cute::cosize_v<GTotalLayout>) * uint32_t(sizeof(float)) +
        uint32_t(cute::cosize_v<LMLayout>) * uint32_t(sizeof(BF16)) * 2;

    // --- warp specialization
    int warp_id = cutlass::canonical_warp_idx_sync();
    bool lane_predicate = cute::elect_one_sync();
    WarpRole warp_role = WarpRole::NonParticipant;
    if (warp_id < kComputeThreads / kWarpSize) {
        warp_role = WarpRole::MMA;
    } else if (warp_id < kComputeThreads / kWarpSize + 1) {
        warp_role = WarpRole::LOAD_QKG;
    } else if (warp_id < kComputeThreads / kWarpSize + 2) {
        warp_role = WarpRole::STORE;
    }

    if constexpr (HasStateIn) {
        if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {
            shared_storage.state_tma_barrier.init(1);
            cutlass::arch::fence_barrier_init();
        }
    }

    using LoadPipelineState = cutlass::PipelineState<InputStages>;
    using LoadPipeline = cutlass::PipelineTmaAsync<InputStages>;
    LoadPipeline load_pipeline = make_load_pipeline<InputStages>(
        shared_storage.load_pipeline,
        kTmaTransactionBytes,
        warp_role, 1, kComputeThreads
    );
    using StorePipelineState = cutlass::PipelineState<OutputStages>;
    using StorePipeline = cutlass::PipelineAsync<OutputStages>;
    StorePipeline store_pipeline = make_store_pipeline<OutputStages>(
        shared_storage.store_pipeline,
        warp_role, kComputeThreads, 1
    );
    // Both pipeline constructors initialize shared barriers. One collective
    // wait covers them and the optional state-load barrier above.
    cutlass::pipeline_init_wait(1);

    // --- per-block sequence info
    int v_idx = int(blockIdx.x) % kVSlices;
    int head_idx = int(blockIdx.x) / kVSlices;
    int seq_idx = int(blockIdx.y);
    if constexpr (IsVarlen) {
        // vLLM orders decodes before prefills. Reverse that order so the
        // long-running prefill CTAs are issued first.
        seq_idx = N - 1 - seq_idx;
    }
    int64_t bos, eos;
    int tile_base;

    if constexpr (IsVarlen) {
        bos = cu_seqlens[seq_idx];
        eos = cu_seqlens[seq_idx + 1];
        // Compute tile_base via linear scan (no host-precomputed table)
        tile_base = 0;
        for (int i = 0; i < seq_idx; i++) {
            tile_base += (int(cu_seqlens[i + 1] - cu_seqlens[i]) + CHUNK - 1) / CHUNK;
        }
    } else {
        int T_seq = T_total / N;
        bos = seq_idx * T_seq;
        eos = bos + T_seq;
        tile_base = seq_idx * ((T_seq + CHUNK - 1) / CHUNK);
    }
    int seq_len  = int(eos - bos);
    int t_tiles  = (seq_len + CHUNK - 1) / CHUNK;

    // --- Load initial state
    if constexpr (HasStateIn) {
        if (warp_role == WarpRole::LOAD_QKG) {
            using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
            constexpr uint32_t kStateTransactionBytes =
                cute::cosize_v<StateSmemLayout> * (StateFP32 ? sizeof(float) : sizeof(BF16));

            Tensor g_init = tma_load_initial_state.get_tma_tensor(make_shape(N * H, D, D));
            auto init_off = g_init.layout()(
                seq_idx * H + head_idx, v_idx * VD, 0);
            Tensor g_init_tile = make_tensor(g_init.data() + init_off,
                make_layout(make_shape(Int<1>{}, Int<VD>{}, Int<D>{}),
                            stride(g_init.layout())));
            auto cta_tma_load_state = tma_load_initial_state.get_slice(Int<0>{});

            if constexpr (StateFP32) {
                using TMAFP32StateSmemLayout = typename Layouts::TMAFP32StateSmemLayout;
                Tensor s_state = make_tensor(
                    make_smem_ptr(reinterpret_cast<float*>(shared_storage.state_staging)),
                    TMAFP32StateSmemLayout{});
                if (cute::elect_one_sync()) {
                    shared_storage.state_tma_barrier.arrive_and_expect_tx(kStateTransactionBytes);
                    cute::copy(
                        tma_load_initial_state.with(reinterpret_cast<BarrierType&>(shared_storage.state_tma_barrier)),
                        cta_tma_load_state.partition_S(g_init_tile),
                        cta_tma_load_state.partition_D(s_state)
                    );
                }
            } else {
                Tensor s_state = make_tensor(
                    make_smem_ptr(reinterpret_cast<BF16*>(shared_storage.state_staging)),
                    TMAStateSmemLayout{});
                if (cute::elect_one_sync()) {
                    shared_storage.state_tma_barrier.arrive_and_expect_tx(kStateTransactionBytes);
                    cute::copy(
                        tma_load_initial_state.with(reinterpret_cast<BarrierType&>(shared_storage.state_tma_barrier)),
                        cta_tma_load_state.partition_S(g_init_tile),
                        cta_tma_load_state.partition_D(s_state)
                    );
                }
            }
            shared_storage.state_tma_barrier.wait(0);
        }
        __syncthreads();
        cutlass::arch::fence_view_async_shared();
    }

    // Load the initial state once into BF16 C fragments; each MMA warp owns
    // VD / 4 value columns and keeps them resident across the chunk loop.
    auto resident_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
        Layout<Shape<_1,_1>>{},
        Tile<_16,_16,_16>{});
    const int resident_warp_id = threadIdx.x / kWarpSize;
    const int resident_lane_id = threadIdx.x % kWarpSize;
    auto resident_thr_mma = resident_mma.get_slice(resident_lane_id);
    Tensor resident_ref = make_tensor(
        make_smem_ptr(reinterpret_cast<BF16*>(shared_storage.state_staging)),
        StateSmemLayout{});
    auto resident_c_ref = resident_thr_mma.partition_C(local_tile(
        resident_ref,
        make_shape(Int<16>{}, Int<16>{}),
        make_coord(0, 0)));
    using ResidentStateFragment = decltype(make_fragment_like<BF16>(
        resident_thr_mma.make_fragment_C(resident_c_ref)));
    constexpr int kResidentStateRowBlocks = D / 16;
    ResidentStateFragment
        resident_state[kValueBlocksPerWarp][kResidentStateRowBlocks];

    if (warp_role == WarpRole::MMA) {
        if constexpr (HasStateIn && !StateFP32) {
            Tensor staged_state_t = make_tensor(
                make_smem_ptr(reinterpret_cast<BF16*>(shared_storage.state_staging)),
                TransposedStateSmemLayout{});
            auto resident_load_c = make_tiled_copy_C(
                Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, resident_mma);
            auto resident_thr_load_c = resident_load_c.get_slice(resident_lane_id);

            #pragma unroll
            for (int m = 0; m < kResidentStateRowBlocks; ++m) {
                #pragma unroll
                for (int bi = 0; bi < kValueBlocksPerWarp; ++bi) {
                    Tensor state_block = local_tile(
                        staged_state_t,
                        make_shape(Int<16>{}, Int<16>{}),
                        make_coord(
                            m, resident_warp_id * kValueBlocksPerWarp + bi));
                    cute::copy(
                        resident_load_c,
                        resident_thr_load_c.partition_S(state_block),
                        resident_thr_load_c.retile_D(resident_state[bi][m]));
                }
            }
        } else if constexpr (HasStateIn) {
            using FP32StateSmemLayout = typename Layouts::FP32StateSmemLayout;
            Tensor staged_fp32 = make_tensor(
                make_smem_ptr(reinterpret_cast<float*>(shared_storage.state_staging)),
                FP32StateSmemLayout{});
            int value_base = resident_warp_id * kValueBlocksPerWarp * 16 +
                resident_lane_id / 4;
            int key_lane = (resident_lane_id % 4) * 2;

            #pragma unroll
            for (int m = 0; m < kResidentStateRowBlocks; ++m) {
                #pragma unroll
                for (int bi = 0; bi < kValueBlocksPerWarp; ++bi) {
                    auto* destination = reinterpret_cast<uint32_t*>(&resident_state[bi][m](0));
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int value = value_base + bi * 16 + (i / 2) * 8;
                        int key = m * 16 + key_lane + (i % 2) * 8;
                        float2 values = *reinterpret_cast<float2 const*>(&staged_fp32(value, key));
                        BF16 low(values.x);
                        BF16 high(values.y);
                        uint32_t packed = uint32_t(low.raw()) | (uint32_t(high.raw()) << 16);
                        SM75_U32x1_MOVM_T::copy(packed, destination[i]);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int m = 0; m < kResidentStateRowBlocks; ++m) {
                #pragma unroll
                for (int bi = 0; bi < kValueBlocksPerWarp; ++bi) {
                    clear(resident_state[bi][m]);
                }
            }
        }
    }

    // The producer may now overwrite initial-state staging with tile 0.
    if constexpr (HasStateIn) {
        __syncthreads();
    }

    // --- LOAD warp: issue TMA loads for v, beta, and workspace intermediates
    if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {
        Tensor g_v = tma_load_v.get_tma_tensor(make_shape(H, T_total, D));
        Tensor g_beta = tma_load_beta.get_tma_tensor(make_shape(H * T_total));

        LoadPipelineState load_write = cutlass::make_producer_start_state<LoadPipeline>();
        auto cta_tma_load_v = tma_load_v.get_slice(Int<0>{});
        auto cta_tma_load_beta = tma_load_beta.get_slice(Int<0>{});
        for (int t = 0; t < t_tiles; ++t) {
            load_pipeline.producer_acquire(load_write);
            using LoadBarrierType = typename LoadPipeline::ProducerBarrierType;
            LoadBarrierType* tma_barrier = load_pipeline.producer_get_barrier(load_write);
            int stage = load_write.index();
            int ws_idx = head_idx * total_tiles + tile_base + t;

            // TMA load v
            auto v_off = g_v.layout()(head_idx, int(bos) + t * CHUNK, v_idx * VD);
            Tensor g_v_tile = make_tensor(g_v.data() + v_off,
                make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<VD>{}), stride(g_v.layout())));
            Tensor s_v_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].v.begin()), TMAVOLayout{});
            cute::copy(tma_load_v.with(*tma_barrier),
                cta_tma_load_v.partition_S(g_v_tile), cta_tma_load_v.partition_D(s_v_tile));

            // TMA load beta (1D)
            int beta_linear = head_idx * T_total + (int(bos) + t * CHUNK);
            int beta_aligned = beta_linear & ~7;
            auto beta_off = g_beta.layout()(beta_aligned);
            Tensor g_beta_tile = make_tensor(g_beta.data() + beta_off, BetaSmemLayout{});
            Tensor s_beta_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].beta.begin()), TMABetaSmemLayout{});
            cute::copy(tma_load_beta.with(*tma_barrier),
                cta_tma_load_beta.partition_S(g_beta_tile), cta_tma_load_beta.partition_D(s_beta_tile));

            // K1 stores byte images of its swizzled shared-memory tensors.
            // K2 uses the same layouts, so raw bulk copies restore them
            // directly without tensor-map coordinate work or repacking.
            cute::SM90_BULK_COPY_G2S::copy(
                ws_kd + int64_t(ws_idx) * (CHUNK * D),
                reinterpret_cast<uint64_t*>(tma_barrier),
                shared_storage.input[stage].k_decayed.begin(),
                int32_t(CHUNK * D * sizeof(BF16)));
            cute::SM90_BULK_COPY_G2S::copy(
                ws_qd + int64_t(ws_idx) * (CHUNK * D),
                reinterpret_cast<uint64_t*>(tma_barrier),
                shared_storage.input[stage].q_decayed.begin(),
                int32_t(CHUNK * D * sizeof(BF16)));
            cute::SM90_BULK_COPY_G2S::copy(
                ws_kr + int64_t(ws_idx) * (CHUNK * D),
                reinterpret_cast<uint64_t*>(tma_barrier),
                shared_storage.input[stage].k_restored.begin(),
                int32_t(CHUNK * D * sizeof(BF16)));
            cute::SM90_BULK_COPY_G2S::copy(
                ws_gt + int64_t(ws_idx) * D,
                reinterpret_cast<uint64_t*>(tma_barrier),
                shared_storage.input[stage].g_total.begin(),
                int32_t(D * sizeof(float)));
            cute::SM90_BULK_COPY_G2S::copy(
                ws_inv + int64_t(ws_idx) * (CHUNK * CHUNK),
                reinterpret_cast<uint64_t*>(tma_barrier),
                shared_storage.input[stage].INV.begin(),
                int32_t(CHUNK * CHUNK * sizeof(BF16)));
            cute::SM90_BULK_COPY_G2S::copy(
                ws_mqk + int64_t(ws_idx) * (CHUNK * CHUNK),
                reinterpret_cast<uint64_t*>(tma_barrier),
                shared_storage.input[stage].Mqk.begin(),
                int32_t(CHUNK * CHUNK * sizeof(BF16)));

            ++load_write;
        }
        load_pipeline.producer_tail(load_write);
    }

    // --- MMA warps
    if (warp_role == WarpRole::MMA) {
        LoadPipelineState load_read;
        StorePipelineState out_write = cutlass::make_producer_start_state<StorePipeline>();
        int compute_tid = threadIdx.x;

        auto run_tile = [&]<bool StoreFinalFP32>(int t) {
            load_pipeline.consumer_wait(load_read);
            int load_stage = load_read.index();
            int out_stage = out_write.index();

            Tensor v_tile = make_tensor(make_smem_ptr(shared_storage.input[load_stage].v.begin()), VOLayout{});
            Tensor beta_tile = make_tensor(make_smem_ptr(shared_storage.input[load_stage].beta.begin()), BetaSmemLayout{});
            int beta_smem_offset = (head_idx * T_total + int(bos) + t * CHUNK) & 7;
            Tensor out_tile = make_tensor(make_smem_ptr(shared_storage.output[out_stage].out.begin()), VOLayout{});

            Tensor k_decayed = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_decayed.begin()), MMALayout{});
            Tensor q_decayed = make_tensor(make_smem_ptr(shared_storage.input[load_stage].q_decayed.begin()), MMALayout{});
            Tensor g_total = make_tensor(make_smem_ptr(shared_storage.input[load_stage].g_total.begin()), GTotalLayout{});
            Tensor INV = make_tensor(make_smem_ptr(shared_storage.input[load_stage].INV.begin()), LMLayout{});
            Tensor Mqk = make_tensor(make_smem_ptr(shared_storage.input[load_stage].Mqk.begin()), LMLayout{});

            // Fused MMA: v_sub, v_beta, U=INV@v, out=q@s, out+=Mqk@U, s_acc_update
            // Each warp handles VD / 4 value columns in 16-column blocks.
            // U stays in registers via SM75_U32x1_MOVM_T (no smem round-trip)
            {
            Tensor k_restored_t = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()), TransposedMMALayout{});

            auto mma = make_tiled_mma(
                MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                Layout<Shape<_1,_1>>{},
                Tile<_16,_16,_16>{}
            );

            const int warp_id = compute_tid / 32;
            const int lane_id = compute_tid % 32;
            const int group_id = (lane_id / 4) % 8;

            ThrMMA thr_mma = mma.get_slice(lane_id);

            // A copy: K_INTER → LDSM_N (for k_decayed, q_decayed, INV, Mqk)
            auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(lane_id);

            // A copy: MN_INTER → LDSM_T (for k_restored_t in Phase 7)
            auto smem_tiled_copy_A_T = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_copy_A_T   = smem_tiled_copy_A_T.get_thread_slice(lane_id);

            // C load/store
            auto smem_tiled_load_C  = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_load_C    = smem_tiled_load_C.get_slice(lane_id);
            auto smem_tiled_store_C = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
            auto smem_thr_store_C   = smem_tiled_store_C.get_slice(lane_id);

            Tensor A_ref = local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor B_ref = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor C_ref = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));

            Tensor tCrAi_k = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_k_view = smem_thr_copy_A.retile_D(tCrAi_k);
            auto tCrA_k = thr_mma.partition_fragment_A(A_ref);

            Tensor tCrAi_q = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_q_view = smem_thr_copy_A.retile_D(tCrAi_q);
            auto tCrA_q = thr_mma.partition_fragment_A(A_ref);

            auto tCrB = thr_mma.partition_fragment_B(B_ref);

            auto tCrC_ref = thr_mma.partition_C(C_ref);

            using AccFragT = decltype(thr_mma.make_fragment_C(tCrC_ref));
            using SFragT = decltype(make_fragment_like<BF16>(thr_mma.make_fragment_C(tCrC_ref)));
            using AFragT = decltype(thr_mma.partition_fragment_A(A_ref));
            using BFragT_u = decltype(thr_mma.partition_fragment_B(B_ref));

            AccFragT u_acc[kValueBlocksPerWarp], out_acc[kValueBlocksPerWarp];
            #pragma unroll
            for (int i = 0; i < kValueBlocksPerWarp; ++i) { u_acc[i] = thr_mma.make_fragment_C(tCrC_ref); clear(u_acc[i]); }
            #pragma unroll
            for (int i = 0; i < kValueBlocksPerWarp; ++i) { out_acc[i] = thr_mma.make_fragment_C(tCrC_ref); clear(out_acc[i]); }

            // ======== Phase 1: Dual GEMM k@s and q@s ========
            constexpr int K_BLOCKS = decltype(cute::size<1>(k_decayed))::value / 16;

            {
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_k_view);
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_q_view);

            #pragma unroll
            for (int k = 0; k < K_BLOCKS; ++k) {
                cute::transform(tCrAi_k, tCrA_k, cute::identity{});
                cute::transform(tCrAi_q, tCrA_q, cute::identity{});
                movm_transpose_c_to_b_16x16(resident_state[0][k], tCrB);
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[0]);
                gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[0]);

                if constexpr (kValueBlocksPerWarp == 2) {
                    movm_transpose_c_to_b_16x16(
                        resident_state[1][k], tCrB);
                }

                if (k + 1 < K_BLOCKS) {
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_k_view);
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_q_view);
                }

                if constexpr (kValueBlocksPerWarp == 2) {
                    gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[1]);
                    gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[1]);
                }
            }
            }

            // ======== Phase 2: Cast out (keep in regs), load v/INV/beta ========
            SFragT out_bf16[kValueBlocksPerWarp];
            #pragma unroll
            for (int i = 0; i < kValueBlocksPerWarp; ++i)
                cute::transform(out_acc[i], out_bf16[i], [] __device__ (float x) { return BF16(x); });

            SFragT v_bf16[kValueBlocksPerWarp];
            #pragma unroll
            for (int i = 0; i < kValueBlocksPerWarp; ++i) {
                Tensor v_block = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, warp_id * kValueBlocksPerWarp + i));
                copy(smem_tiled_load_C, smem_thr_load_C.partition_S(v_block), smem_thr_load_C.retile_D(v_bf16[i]));
            }

            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(INV), tCrAi_k_view);
            cute::transform(tCrAi_k, tCrA_k, cute::identity{});

            BF16 beta0 = BF16(sigmoid_tanh_approx_f32(float(beta_tile(beta_smem_offset + group_id))));
            BF16 beta1 = BF16(sigmoid_tanh_approx_f32(float(beta_tile(beta_smem_offset + group_id + 8))));

            // ======== Phase 3: u = (v - u) * beta; u = INV @ u (per block) ========
            SFragT u_bf16[kValueBlocksPerWarp];
            uint32_t u_b_regs[4];

            #pragma unroll
            for (int i = 0; i < kValueBlocksPerWarp; ++i) {
                cute::transform(u_acc[i], u_bf16[i], [] __device__ (float x) { return BF16(x); });

                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        auto c0 = make_coord(make_coord(a, 0), 0, d);
                        auto c1 = make_coord(make_coord(a, 1), 0, d);
                        u_bf16[i](c0) = (v_bf16[i](c0) - u_bf16[i](c0)) * beta0;
                        u_bf16[i](c1) = (v_bf16[i](c1) - u_bf16[i](c1)) * beta1;
                    }
                }

                uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16[i](0));
                SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]);
                SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
                SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
                SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);

                auto tCrB_u_tmp = thr_mma.partition_fragment_B(B_ref);
                uint32_t* b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_tmp(0));
                b_dst[0] = u_b_regs[0]; b_dst[1] = u_b_regs[1];
                b_dst[2] = u_b_regs[2]; b_dst[3] = u_b_regs[3];

                clear(u_acc[i]);
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_tmp(_,_,Int<0>{}), u_acc[i]);

                cute::transform(u_acc[i], u_bf16[i], [] __device__ (float x) { return BF16(x); });
            }

            // ======== Phase 4: Load Mqk, MOVM_T → tCrB_u_arr, Mqk@U + add out ========
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(Mqk), tCrAi_k_view);
            cute::transform(tCrAi_k, tCrA_k, cute::identity{});

            BFragT_u tCrB_u_arr[kValueBlocksPerWarp];

            #pragma unroll
            for (int i = 0; i < kValueBlocksPerWarp; ++i) {
                uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16[i](0));
                SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]);
                SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
                SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
                SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);

                tCrB_u_arr[i] = thr_mma.partition_fragment_B(B_ref);
                uint32_t* b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_arr[i](0));
                b_dst[0] = u_b_regs[0]; b_dst[1] = u_b_regs[1];
                b_dst[2] = u_b_regs[2]; b_dst[3] = u_b_regs[3];

                clear(out_acc[i]);
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_arr[i](_,_,Int<0>{}), out_acc[i]);

                SFragT gemm_bf16;
                cute::transform(out_acc[i], gemm_bf16, [] __device__ (float x) { return BF16(x); });
                cute::transform(out_bf16[i], gemm_bf16, out_bf16[i], [] __device__ (BF16 c, BF16 a) { return c + a; });
            }

            // Late output-acquire (cycle trim): first out-stage write is in
            // phase 5, so ownership is needed only now; the wait typically
            // finds the stage already released.
            store_pipeline.producer_acquire(out_write);
            // ======== Phase 5: Store final out ========
            #pragma unroll
            for (int i = 0; i < kValueBlocksPerWarp; ++i) {
                Tensor out_block = local_tile(out_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, warp_id * kValueBlocksPerWarp + i));
                copy(smem_tiled_store_C, smem_thr_store_C.retile_S(out_bf16[i]), smem_thr_store_C.partition_D(out_block));
            }

            // ======== Phase 6: s_acc update ========
            // s_acc[D, VD] = s_acc * g_total +
            //                 k_restored_t[D, 16] @ U[16, VD]
            // Each warp updates its VD/4 value columns.
            // U is already in tCrB_u_arr[0..1] as B operands (from Phase 4 MOVM_T)
            constexpr int PREFETCH = 1;
            constexpr int S_M_BLOCKS = decltype(cute::size<0>(k_restored_t))::value / 16;

            Tensor tCrAi_kr = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_kr_view = smem_thr_copy_A_T.retile_D(tCrAi_kr);

            AFragT ring_A_kr[PREFETCH];
            float ring_g0[PREFETCH], ring_g1[PREFETCH];
            auto state_coord = make_identity_tensor(make_shape(Int<16>{}, Int<16>{}));
            auto tCcState = thr_mma.partition_C(state_coord);
            int state_idx = seq_idx * H + head_idx;
            auto* final_state = static_cast<float*>(final_state_raw_ptr);

            #pragma unroll
            for (int i = 0; i < PREFETCH; ++i) {
                Tensor kr_block = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(i, 0));
                copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_block), tCrAi_kr_view);
                cute::transform(tCrAi_kr, ring_A_kr[i], cute::identity{});

                ring_g0[i] = g_total(i * 16 + group_id);
                ring_g1[i] = g_total(i * 16 + group_id + 8);
            }

            #pragma unroll
            for (int m = 0; m < S_M_BLOCKS; ++m) {
                const int slot = m % PREFETCH;

                float g0 = ring_g0[slot];
                float g1 = ring_g1[slot];

                #pragma unroll
                for (int bi = 0; bi < kValueBlocksPerWarp; ++bi) {
                    clear(u_acc[bi]);
                    gemm(thr_mma, ring_A_kr[slot](_,_,Int<0>{}), tCrB_u_arr[bi](_,_,Int<0>{}), u_acc[bi]);
                }

                if (m + PREFETCH < S_M_BLOCKS) {
                    Tensor kr_next = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(m + PREFETCH, 0));
                    copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_next), tCrAi_kr_view);
                    cute::transform(tCrAi_kr, ring_A_kr[slot], cute::identity{});

                    ring_g0[slot] = g_total((m + PREFETCH) * 16 + group_id);
                    ring_g1[slot] = g_total((m + PREFETCH) * 16 + group_id + 8);
                }

                #pragma unroll
                for (int bi = 0; bi < kValueBlocksPerWarp; ++bi) {
                    auto& state_fragment = resident_state[bi][m];
                    int value_base = v_idx * VD + (resident_warp_id * kValueBlocksPerWarp + bi) * 16;

                    #pragma unroll
                    for (int a = 0; a < 2; ++a) {
                        #pragma unroll
                        for (int d = 0; d < 2; ++d) {
                            auto c0 = make_coord(make_coord(a, 0), 0, d);
                            auto c1 = make_coord(make_coord(a, 1), 0, d);
                            float h0 = bf16_to_f32(state_fragment(c0)) * g0 + u_acc[bi](c0);
                            float h1 = bf16_to_f32(state_fragment(c1)) * g1 + u_acc[bi](c1);
                            if constexpr (StoreFinalFP32) {
                                // Export raw FP32 before rounding resident H.
                                auto coord0 = tCcState(c0);
                                auto coord1 = tCcState(c1);
                                int key0 = m * 16 + int(get<0>(coord0));
                                int key1 = m * 16 + int(get<0>(coord1));
                                int value0 = value_base + int(get<1>(coord0));
                                int value1 = value_base + int(get<1>(coord1));
                                int64_t offset0 = (int64_t(state_idx) * D + value0) * D + key0;
                                int64_t offset1 = (int64_t(state_idx) * D + value1) * D + key1;
                                final_state[offset0] = h0;
                                final_state[offset1] = h1;
                            }
                            state_fragment(c0) = BF16(h0);
                            state_fragment(c1) = BF16(h1);
                        }
                    }
                }
            }
            }
            // The collective commit releases this input stage and publishes
            // the output tile.
            cutlass::arch::fence_view_async_shared();
            store_pipeline.producer_commit(out_write);
            load_pipeline.consumer_release(load_read);
            ++load_read;
            ++out_write;
        };

        // Peel the final tile so steady-state iterations contain no H stores.
        if constexpr (HasStateOut && StateFP32) {
            for (int t = 0; t + 1 < t_tiles; ++t) {
                run_tile.template operator()<false>(t);
            }
            if (t_tiles > 0) {
                run_tile.template operator()<true>(t_tiles - 1);
            }
        } else {
            for (int t = 0; t < t_tiles; ++t) {
                run_tile.template operator()<false>(t);
            }
        }

        if constexpr (HasStateOut && !StateFP32) {
            int state_idx = seq_idx * H + head_idx;
            int value_base = v_idx * VD +
                resident_warp_id * kValueBlocksPerWarp * 16 +
                resident_lane_id / 4;
            int key_lane = (resident_lane_id % 4) * 2;

            #pragma unroll
            for (int m = 0; m < kResidentStateRowBlocks; ++m) {
                #pragma unroll
                for (int bi = 0; bi < kValueBlocksPerWarp; ++bi) {
                    auto const* source = reinterpret_cast<uint32_t const*>(&resident_state[bi][m](0));

                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        uint32_t transposed;
                        // MOVM makes this a contiguous K pair in H[V,K].
                        SM75_U32x1_MOVM_T::copy(source[i], transposed);
                        int value = value_base + bi * 16 + (i / 2) * 8;
                        int key = m * 16 + key_lane + (i % 2) * 8;
                        int64_t offset = (int64_t(state_idx) * D + value) * D + key;

                        *reinterpret_cast<uint32_t*>(static_cast<BF16*>(final_state_raw_ptr) + offset) = transposed;
                    }
                }
            }
        }
    }

    if constexpr (HasStateOut && StateFP32) {
        // Empty sequences skip the MMA loop; the store warp exports H0.
        if (warp_role == WarpRole::STORE && t_tiles == 0) {
            int state_idx = seq_idx * H + head_idx;
            int64_t offset =
                (int64_t(state_idx) * D + v_idx * VD) * D;
            auto* final_state =
                static_cast<float2*>(final_state_raw_ptr) + offset / 2;

            for (int i = resident_lane_id; i < VD * D / 2;
                 i += kWarpSize) {
                float2 state = make_float2(0.0f, 0.0f);
                if constexpr (HasStateIn) {
                    using FP32StateSmemLayout =
                        typename Layouts::FP32StateSmemLayout;
                    Tensor staged_state = make_tensor(
                        make_smem_ptr(reinterpret_cast<float*>(
                            shared_storage.state_staging)),
                        FP32StateSmemLayout{});
                    int value = i / (D / 2);
                    int key = i % (D / 2) * 2;
                    state.x = staged_state(value, key);
                    state.y = staged_state(value, key + 1);
                    state.x = bf16_to_f32(BF16(state.x));
                    state.y = bf16_to_f32(BF16(state.y));
                }
                final_state[i] = state;
            }
        }
    }

    if (warp_role == WarpRole::STORE && lane_predicate) {
        Tensor g_out = tma_store_out.get_tma_tensor(make_shape(H, T_total, D));
        auto cta_tma_store = tma_store_out.get_slice(Int<0>{});
        StorePipelineState out_read;
        for (int t = 0; t < t_tiles; ++t) {
            store_pipeline.consumer_wait(out_read);
            int stage = out_read.index();
            int actual_len = min(CHUNK, seq_len - t * CHUNK);

            BF16* out_stage_ptr = shared_storage.output[stage].out.begin();

            if (actual_len < CHUNK) {
                // Manual store for tail tile to avoid overwriting next sequence
                // Only one thread runs here, so loop over this V slice.
                Tensor s_out = make_tensor(make_smem_ptr(out_stage_ptr), VOLayout{});
                for (int row = 0; row < actual_len; ++row) {
                    int64_t global_base = (bos + t * CHUNK + row) * H * D + head_idx * D + v_idx * VD;
                    for (int col = 0; col < VD; ++col) {
                        out_raw_ptr[global_base + col] = s_out(row, col);
                    }
                }
            } else {
                // TMA store for full tiles
                auto out_off = g_out.layout()(head_idx, int(bos) + t * CHUNK, v_idx * VD);
                Tensor g_out_tile = make_tensor(g_out.data() + out_off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<VD>{}), stride(g_out.layout())));
                Tensor s_out_tile = make_tensor(make_smem_ptr(out_stage_ptr), TMAVOLayout{});
                cute::copy(
                    tma_store_out,
                    cta_tma_store.partition_S(s_out_tile),
                    cta_tma_store.partition_D(g_out_tile)
                );
                tma_store_arrive();
            }

            tma_store_wait<0>();
            store_pipeline.consumer_release(out_read);
            ++out_read;
        }

    }

}
