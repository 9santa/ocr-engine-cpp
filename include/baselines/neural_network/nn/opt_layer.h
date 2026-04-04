#pragma once
#include "baselines/neural_network/nn/opt_neuron.h"

struct OptLayer {
    std::vector<OptNeuron> neurons;

    OptLayer(int nin, int nout, bool nonlin = true);

    std::vector<OptValPtr> operator()(GraphArena& arena, const std::vector<OptValPtr>& x);
    std::vector<OptValPtr> parameters();
    void zero_grad();
};
