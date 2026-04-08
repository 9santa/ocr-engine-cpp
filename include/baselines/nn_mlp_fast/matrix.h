#pragma once

#include <vector>


/* Matrix 'container' for:
     - batch input X
     - logits
     - hidden activations
     - gradients */
struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<float> data;

    Matrix() = default;
    Matrix(int r, int c, float value = 0.0f);

    float& operator()(int r, int c);
    const float& operator()(int r, int c) const;

    void fill(float value);
};
