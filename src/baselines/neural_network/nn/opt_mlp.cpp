#include "baselines/neural_network/nn/opt_mlp.h"

OptMLP::OptMLP(int nin, const std::vector<int>& nouts) {
    std::vector<int> sz = {nin};
    sz.insert(sz.end(), nouts.begin(), nouts.end());

    for (size_t i = 0; i < nouts.size(); i++) {
        bool nonlin = (i + 1 < nouts.size());
        layers.emplace_back(sz[i], sz[i+1], nonlin);
    }

    rebuild_param_cache();
}

void OptMLP::rebuild_param_cache() {
    flat_params.clear();
    for (auto& layer : layers) {
        for (auto& neuron : layer.neurons) {
            flat_params.insert(flat_params.end(), neuron.w.begin(), neuron.w.end());
            flat_params.push_back(neuron.b);
        }
    }
}

std::vector<OptValPtr> OptMLP::operator()(GraphArena& arena, const std::vector<OptValPtr>& x) {
    std::vector<OptValPtr> out = x;
    for (auto& layer : layers) out = layer(arena, out);
    return out;
}

const std::vector<OptValPtr>& OptMLP::parameters() {
    return flat_params;
}

void OptMLP::zero_grad() {
    for (auto& layer : layers) layer.zero_grad();
}
