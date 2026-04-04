#include "baselines/neural_network/nn/loss.h"

#include <algorithm>
#include <cmath>

std::vector<OptValPtr> softmax(GraphArena& arena, const std::vector<OptValPtr>& logits) {
    double maxLogit = -1e9;
    for (auto& l : logits) maxLogit = std::max(maxLogit, l->data);

    std::vector<double> exps(logits.size());
    double sumExp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        exps[i] = std::exp(logits[i]->data - maxLogit);
        sumExp += exps[i];
    }

    std::vector<OptValPtr> probs(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        probs[i] = arena.make_value(exps[i] / sumExp);
    }

    return probs;
}

OptValPtr cross_entropy_loss(GraphArena& arena, const std::vector<OptValPtr> &logits, int true_label) {
    auto probs = softmax(arena, logits);
    double p = probs[true_label]->data;
    auto loss = arena.make_value(-std::log(p));

    // Gradients for backward pass
    for (size_t i = 0; i < probs.size(); i++) {
        double grad = probs[i]->data;
        if ((int)i == true_label) grad -= 1.0;
        logits[i]->grad = grad;
    }

    return loss;
}
