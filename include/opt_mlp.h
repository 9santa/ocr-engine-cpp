#pragma once
#include "opt_layer.h"

struct OptMLP {
    std::vector<OptLayer> layers;

    explicit OptMLP(int nin, const std::vector<int>& nouts);

    std::vector<OptValPtr> operator()(const std::vector<OptValPtr>& x);
    std::vector<OptValPtr> parameters();
    void zero_grad();
};
