#pragma once
#include "baselines/neural_network/nn/opt_value.h"
#include <memory>
#include <vector>

struct OptNeuron {
    std::vector<std::unique_ptr<OptValue>> owned_w;
    std::unique_ptr<OptValue> owned_b;

    std::vector<OptValPtr> w;
    OptValPtr b;
    bool nonlin;

    explicit OptNeuron(int nin, bool nonlin=true);
    OptValPtr operator()(GraphArena& arena, const std::vector<OptValPtr>& x);
    std::vector<OptValPtr> parameters();
    void zero_grad();
};
