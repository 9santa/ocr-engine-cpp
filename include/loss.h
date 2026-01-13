#pragma once
#include "opt_value.h"
#include <vector>

OptValPtr cross_entropy_loss(const std::vector<OptValPtr>& logits, int true_label);
std::vector<OptValPtr> softmax(const std::vector<OptValPtr>& logits);


