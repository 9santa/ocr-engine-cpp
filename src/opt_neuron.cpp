#include <random>
#include "opt_neuron.h"

OptNeuron::OptNeuron(int nin, bool _nonlin) : nonlin(_nonlin) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < nin; i++) w.push_back(new OptValue(dis(gen)));
    b = new OptValue(0.0);
}

OptValPtr OptNeuron::operator()(const std::vector<OptValPtr>& x) {
    OptValPtr act = b;
    for (size_t i = 0; i < w.size(); i++) {
        act = add(act, mul(w[i], x[i]));
    }
    return nonlin ? tanh(act) : act;
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
