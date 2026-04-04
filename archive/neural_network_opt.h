#pragma once

#include "gradient_opt.h"
#include <vector>

class OptNeuron {
public:
    std::vector<OptValPtr> w;
    OptValPtr b;
    bool nonlin;

    OptNeuron(int nin, bool nonlin=true);
    OptValPtr operator()(const std::vector<OptValPtr>& x);
    std::vector<OptValPtr> parameters();

    void zero_grad();
};

class OptLayer {
public:
    std::vector<OptNeuron> neurons;
    int nin, nout;

    OptLayer(int nin, int nout);
    std::vector<OptValPtr> operator()(const std::vector<OptValPtr>& x);
    std::vector<OptValPtr> parameters();

    void zero_grad();
};

class OptMLP {
public:
    std::vector<OptLayer> layers;

    OptMLP(int nin, const std::vector<int>& nouts);
    std::vector<OptValPtr> operator()(const std::vector<OptValPtr>& x);
    std::vector<OptValPtr> parameters();

    void zero_grad();
};

OptValPtr mse_opt(const std::vector<OptValPtr>& ys, const std::vector<OptValPtr>& ypred);
