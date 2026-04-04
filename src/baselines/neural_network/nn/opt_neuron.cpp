#include <random>

#include "baselines/neural_network/nn/opt_neuron.h"
#include "baselines/neural_network/nn/opt_ops.h"

OptNeuron::OptNeuron(int nin, bool _nonlin) : nonlin(_nonlin) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    owned_w.reserve(nin);
    w.reserve(nin);

    for (int i = 0; i < nin; i++) {
        owned_w.push_back(std::make_unique<OptValue>(dis(gen)));
        w.push_back(owned_w.back().get());
    }

    owned_b = std::make_unique<OptValue>(0.0);
    b = owned_b.get();
}

OptValPtr OptNeuron::operator()(GraphArena& arena, const std::vector<OptValPtr>& x) {
    OptValPtr act = b;
    for (size_t i = 0; i < w.size(); i++) {
        // act = act + w[i] * x[i]
        auto wx = mul(arena, w[i], x[i]);
        act = add(arena, act, wx);
    }
    return nonlin ? tanh(arena, act) : act;
}

std::vector<OptValPtr> OptNeuron::parameters() {
    auto params = w;
    params.push_back(b);
    return params;
}

void OptNeuron::zero_grad() {
    for (auto p : w) p->zero_grad();
    b->zero_grad();
}
