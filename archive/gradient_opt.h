#pragma once

#include <memory>
#include <vector>
#include <functional>


class OptValue;
using OptValPtr = std::shared_ptr<OptValue>;

extern std::vector<std::function<void()>> gradient_ops;

class OptValue : public std::enable_shared_from_this<OptValue> {
public:
    double data;
    double grad;

    // not storing parents, backward functions
    explicit OptValue(double data, double grad = 0.0) : data(data), grad(grad) {}

    // operations
    OptValPtr operator+(const OptValPtr& other);
    OptValPtr operator-(const OptValPtr& other);
    OptValPtr operator*(const OptValPtr& other);
    OptValPtr tanh() const;

    static void apply_gradients();

    inline void zero_grad() { grad = 0.0; };

};

// factory function
OptValPtr make_value(double data);
