#pragma once

#include "baselines/nn_mlp_fast/matrix.h"
#include <vector>

struct SoftmaxCrossEntropyResult {
    float loss = 0.0f;
    Matrix probs;
};

SoftmaxCrossEntropyResult softmaxCrossEntropyForward(const Matrix& logits, const std::vector<int>& labels);

Matrix softmaxCrossEntropyBackward(const Matrix& probs, const std::vector<int>& labels);
