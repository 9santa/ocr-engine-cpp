#pragma once

#include <vector>

struct TrainingSample {
    std::vector<float> features;
    int label;
};
