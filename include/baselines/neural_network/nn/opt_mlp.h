#pragma once
#include "baselines/neural_network/nn/opt_layer.h"

struct OptMLP {
    std::vector<OptLayer> layers;
    std::vector<OptValPtr> flat_params;

    explicit OptMLP(int nin, const std::vector<int>& nouts);

    void rebuild_param_cache();
    std::vector<OptValPtr> operator()(GraphArena& arena, const std::vector<OptValPtr>& x);
    const std::vector<OptValPtr>& parameters();
    void zero_grad();
};
