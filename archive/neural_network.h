#pragma once
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "gradient.h"
#include <vector>
#include <memory>

class Neuron {
public:
    std::vector<ValPtr> w;
    ValPtr b;
    bool nonlin;

    Neuron(int nin, bool nonlin = true);
    ValPtr operator()(const std::vector<ValPtr>& x);
    std::vector<ValPtr> parameters();
};

class Layer {
public:
    std::vector<Neuron> neurons;
    int nin, nout;

    Layer(int nin, int nout);
    std::vector<ValPtr> operator()(const std::vector<ValPtr>& x);
    std::vector<ValPtr> parameters();
};

class MLP {
public:
    std::vector<Layer> layers;

    MLP(int nin, const std::vector<int>& nouts);
    std::vector<ValPtr> operator()(const std::vector<ValPtr>& x);
    std::vector<ValPtr> parameters();
};

// Declare functions (don't define them here)
ValPtr MSE(const std::vector<ValPtr>& ys, const std::vector<ValPtr>& ypred);
void train(MLP& l, const std::vector<std::vector<ValPtr>>& input, const std::vector<ValPtr>& ys);

#endif // NEURAL_NETWORK_H
