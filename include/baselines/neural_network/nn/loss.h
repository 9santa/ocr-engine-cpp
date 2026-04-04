#pragma once
#include "baselines/neural_network/nn/opt_value.h"
#include <vector>

OptValPtr cross_entropy_loss(GraphArena& arena, const std::vector<OptValPtr>& logits, int true_label);
std::vector<OptValPtr> softmax(GraphArena& arena, const std::vector<OptValPtr>& logits);

