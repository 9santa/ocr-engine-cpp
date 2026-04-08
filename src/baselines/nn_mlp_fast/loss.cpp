#include "baselines/nn_mlp_fast/loss.h"

#include <algorithm>
#include <cmath>

SoftmaxCrossEntropyResult softmaxCrossEntropyForward(const Matrix &logits, const std::vector<int> &labels) {
    SoftmaxCrossEntropyResult result;
    result.probs = Matrix(logits.rows, logits.cols, 0.0f);

    float totalLoss = 0.0f;

    for (int r = 0; r < logits.rows; r++) {
        float maxLogit = logits(r, 0);
        for (int c = 1; c < logits.cols; c++) {
            maxLogit = std::max(maxLogit, logits(r, c));
        }

        float sumExp = 0.0f;
        for (int c = 0; c < logits.cols; c++) {
            float e = std::exp(logits(r, c) - maxLogit);
            result.probs(r, c) = e;
            sumExp += e;
        }

        for (int c = 0; c < logits.cols; c++) {
            result.probs(r, c) /= sumExp;
        }

        float p = result.probs(r, labels[r]);
        totalLoss += -std::log(std::max(p, 1e-8f));
    }

    result.loss = totalLoss / static_cast<float>(logits.rows);
    return result;
}

Matrix softmaxCrossEntropyBackward(const Matrix &probs, const std::vector<int> &labels) {
    Matrix grad = probs;

    for (int r = 0; r < grad.rows; r++) {
        grad(r, labels[r]) -= 1.0f;
    }

    const float scale = 1.0f / static_cast<float>(grad.rows);
    for (auto& v : grad.data) {
        v *= scale;
    }

    return grad;
}
