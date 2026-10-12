#pragma once
#include <cuda_runtime.h>

#include <cutlass/bfloat16.h>

template <
    int D,
    bool HasStateIn = true,
    bool HasStateOut = true,
    bool StateFP32 = false,
    bool HasCheckpoint = false,
    bool IsVarlen = true,
    typename SeqlenT = int64_t>
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
    SeqlenT const* checkpoint_offsets_ptr,
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
    int num_sms,
    cudaStream_t stream
);

// The two kernels of launch_fwd as standalone launches. launch_prepare (K1)
// fills ``workspace`` from q, k, g and beta; launch_recurrence (K2) reads
// that workspace with v and beta and writes ``out`` and the states. Both
// must use the same geometry (total_tiles, T_total, H, N, cu_seqlens).
template <
    int D,
    bool IsVarlen = true,
    typename SeqlenT = int64_t>
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
    SeqlenT const* cu_seqlens_ptr,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    cudaStream_t stream
);

template <
    int D,
    bool HasStateIn = true,
    bool HasStateOut = true,
    bool StateFP32 = false,
    bool HasCheckpoint = false,
    bool IsVarlen = true,
    typename SeqlenT = int64_t>
void launch_recurrence(
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    void const* initial_state_ptr,
    void* final_state_ptr,
    void* checkpoint_state_ptr,
    SeqlenT const* checkpoint_offsets_ptr,
    cutlass::bfloat16_t* out_ptr,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    SeqlenT const* cu_seqlens_ptr,
    int num_sms,
    cudaStream_t stream
);
