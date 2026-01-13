#pragma once
#include "opt_value.h"
#include "opt_ops.h"
#include <vector>

struct OptNeuron {
    std::vector<OptValPtr> w; // weights
    OptValPtr b; // bias
    bool nonlin;

    explicit OptNeuron(int nin, bool nonlin=true);

    OptValPtr operator()(const std::vector<OptValPtr>& x);

    std::vector<OptValPtr> parameters();
    void zero_grad();
};
