#pragma once

#include "baselines/nn_mlp_fast/matrix.h"
#include <vector>

class DenseLayer {
public:
    DenseLayer(int inFeatures, int outFeatures);

    Matrix forward(const Matrix& x);
    Matrix backward(const Matrix& gradOutput);

    void step(float learningRate);
    void zeroGrad();

    int inFeatures() const { return inDim; }
    int outFeatures() const { return outDim; }

private:
    int inDim;
    int outDim;

    Matrix W;
    Matrix dW;

    std::vector<float> b;
    std::vector<float> db;

    Matrix inputCache;
};
