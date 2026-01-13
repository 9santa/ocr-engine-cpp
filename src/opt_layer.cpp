#include "opt_layer.h"

OptLayer::OptLayer(int nin, int nout) {
    for (int i = 0; i < nout; i++) neurons.emplace_back(nin);
}

std::vector<OptValPtr> OptLayer::operator()(const std::vector<OptValPtr>& x) {
    std::vector<OptValPtr> out;
    for (auto& n : neurons) out.push_back(n(x));
    return out;
}

std::vector<OptValPtr> OptLayer::parameters() {
    std::vector<OptValPtr> out;
    for (auto& n : neurons) {
        auto p = n.parameters();
        out.insert(out.end(), p.begin(), p.end());
    }
    return out;
}

void OptLayer::zero_grad() {
    for (auto& n : neurons) n.zero_grad();
}
