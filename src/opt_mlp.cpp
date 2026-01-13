#include "opt_mlp.h"

OptMLP::OptMLP(int nin, const std::vector<int>& nouts) {
    std::vector<int> sz = {nin};
    sz.insert(sz.end(), nouts.begin(), nouts.end());

    for (size_t i = 0; i < nouts.size(); i++) layers.emplace_back(sz[i], sz[i+1]);
}

std::vector<OptValPtr> OptMLP::operator()(const std::vector<OptValPtr>& x) {
    std::vector<OptValPtr> out = x;
    for (auto& layer : layers) out = layer(out);
    return out;
}

std::vector<OptValPtr> OptMLP::parameters() {
    std::vector<OptValPtr> out;
    for (auto& layer : layers) {
        auto p = layer.parameters();
        out.insert(out.end(), p.begin(), p.end());
    }
    return out;
}

void OptMLP::zero_grad() {
    for (auto& layer : layers) layer.zero_grad();
}
