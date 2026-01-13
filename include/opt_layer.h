#pragma once
#include "opt_neuron.h"

struct OptLayer {
    std::vector<OptNeuron> neurons;

    explicit OptLayer(int nin, int nout);

    std::vector<OptValPtr> operator()(const std::vector<OptValPtr>& x);
    std::vector<OptValPtr> parameters();
    void zero_grad();
};
