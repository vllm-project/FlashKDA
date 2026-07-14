#pragma once

#include <optional>

#include <torch/extension.h>

int64_t get_workspace_size(int64_t T_total, int64_t H, int64_t N = 1);

void fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    double scale,
    torch::Tensor out,
    torch::Tensor workspace,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    double lower_bound,
    std::optional<torch::Tensor> initial_state = std::nullopt,
    std::optional<torch::Tensor> final_state = std::nullopt,
    std::optional<torch::Tensor> cu_seqlens = std::nullopt);
