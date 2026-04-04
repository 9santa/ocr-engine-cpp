#include "neural_network_opt.h"
#include "gradient_opt.h"
#include <memory>
#include <random>


OptNeuron::OptNeuron(int nin, bool nonlin) : nonlin(nonlin) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < nin; i++) {
        w.push_back(make_value(dis(gen)));
    }
    b = make_value(0.0);
}

OptValPtr OptNeuron::operator()(const std::vector<OptValPtr>& x) {
    OptValPtr act = b;
    for (size_t i = 0; i < w.size(); i++) {
        auto wx = *w[i] * x[i];
        act = *act + wx;
    }
    return nonlin ? act->tanh() : act;
}

std::vector<OptValPtr> OptNeuron::parameters() {
    std::vector<OptValPtr> params = w;
    params.push_back(b);
    return params;
}

void OptNeuron::zero_grad() {
    for (auto& weight : w) weight->zero_grad();
    b->zero_grad();
}

OptLayer::OptLayer(int nin, int nout) : nin(nin), nout(nout) {
    for (int i = 0; i < nout; i++) {
        neurons.emplace_back(nin);
    }
}

std::vector<OptValPtr> OptLayer::operator()(const std::vector<OptValPtr>& x) {
    std::vector<OptValPtr> outputs;
    for (auto& neuron : neurons) {
        outputs.push_back(neuron(x));
    }
    return outputs;
}

std::vector<OptValPtr> OptLayer::parameters() {
    std::vector<OptValPtr> params;
    for (auto& neuron : neurons) {
        auto neuron_params = neuron.parameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }
    return params;
}

void OptLayer::zero_grad() {
    for (auto& neuron : neurons) {
        neuron.zero_grad();
    }
}

OptMLP::OptMLP(int nin, const std::vector<int>& nouts) {
    std::vector<int> sizes = {nin};
    sizes.insert(sizes.end(), nouts.begin(), nouts.end());

    for (size_t i = 0; i < sizes.size()-1; i++) {
        layers.emplace_back(sizes[i], sizes[i+1]);
    }
}

std::vector<OptValPtr> OptMLP::operator()(const std::vector<OptValPtr>& x) {
    auto output = x;

    for (auto& layer : layers) {
        output = layer(output);
    }
    return output;
}

std::vector<OptValPtr> OptMLP::parameters() {
    std::vector<OptValPtr> params;
    for (auto& l : layers) {
        std::vector<OptValPtr> lp = l.parameters();
        params.insert(params.end(), lp.begin(), lp.end());
    }
    return params;
}

void OptMLP::zero_grad() {
    for (auto& layer : layers) {
        layer.zero_grad();
    }
}


OptValPtr mse_opt(const std::vector<OptValPtr>& ys, const std::vector<OptValPtr>& ypred) {
    auto loss = make_value(0.0);
    for (size_t i = 0; i < ys.size(); i++) {
        auto diff = *ys[i] - ypred[i];
        auto sq = *diff * diff;
        loss = *loss + sq;
    }
    loss = *loss * make_value(1.0 / ys.size());
    return loss;
}
