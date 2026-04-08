#include "baselines/nn_mlp_fast/dense_layer.h"
#include "baselines/nn_mlp_fast/matrix.h"

#include <algorithm>
#include <random>

DenseLayer::DenseLayer(int inFeatures, int outFeatures)
    : inDim(inFeatures),
      outDim(outFeatures),
      W(inFeatures, outFeatures),
      dW(inFeatures, outFeatures, 0.0f),
      b(outFeatures, 0.0f),
      db(outFeatures, 0.0f) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    for (auto& w : W.data) {
        w = dist(gen);
    }
}

Matrix DenseLayer::forward(const Matrix& x) {
    inputCache = x;
    Matrix out(x.rows, outDim, 0.0f);

    for (int r = 0; r < x.rows; r++) {
        for (int j = 0; j < outDim; j++) {
            float sum = b[j];
            for (int i = 0; i < inDim; i++) {
                sum += x(r, i) * W(i, j);
            }
            out(r, j) = sum;
        }
    }

    return out;
}

Matrix DenseLayer::backward(const Matrix& gradOutput) {
    Matrix gradInput(inputCache.rows, inDim, 0.0f);

    dW.fill(0.0f);
    std::fill(db.begin(), db.end(), 0.0f);

    for (int r = 0; r < inputCache.rows; r++) {
        for (int j = 0; j < outDim; j++) {
            const float go = gradOutput(r, j);
            db[j] += go;

            for (int i = 0; i < inDim; i++) {
                dW(i, j) += inputCache(r, i) * go;
                gradInput(r, i) += go * W(i, j);
            }
        }
    }

    return gradInput;
}

void DenseLayer::step(float learningRate) {
    for (int i = 0; i < inDim; i++) {
        for (int j = 0; j < outDim; j++) {
            W(i, j) -= learningRate * dW(i, j);
        }
    }

    for (int j = 0; j < outDim; j++) {
        b[j] -= learningRate * db[j];
    }
}

void DenseLayer::zeroGrad() {
    dW.fill(0.0f);
    std::fill(db.begin(), db.end(), 0.0f);
}
