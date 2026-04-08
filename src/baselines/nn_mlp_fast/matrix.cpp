#include "baselines/nn_mlp_fast/matrix.h"

Matrix::Matrix(int r, int c, float value) : rows(r), cols(c), data(r * c, value) {}

float& Matrix::operator()(int r, int c) {
    return data[r * cols + c];
}

const float& Matrix::operator()(int r, int c) const {
    return data[r * cols + c];
}

void Matrix::fill(float value) {
    std::fill(data.begin(), data.end(), value);
}
