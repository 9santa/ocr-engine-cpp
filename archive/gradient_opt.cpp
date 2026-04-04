#include "gradient_opt.h"
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

std::vector<std::function<void()>> gradient_ops;

OptValPtr make_value(double data) {
    return std::make_shared<OptValue>(data);
}

OptValPtr OptValue::operator+(const OptValPtr& other)  {
    auto out = make_value(data + other->data);

    // store gradient
    gradient_ops.push_back([self=shared_from_this(), other, out]() {
            self->grad += out->grad;
            other->grad += out->grad;
    });

    return out;
}

OptValPtr OptValue::operator-(const OptValPtr& other) {
    auto out = make_value(data - other->data);

    gradient_ops.push_back([self=shared_from_this(), other, out]() {
            self->grad += out->grad;
            other->grad += -out->grad;
    });

    return out;
}

OptValPtr OptValue::operator*(const OptValPtr& other) {
    auto out = make_value(data * other->data);

    gradient_ops.push_back([self=shared_from_this(), other, out]() {
            self->grad += other->data * out->grad;
            other->grad += self->data * out->grad;
    });

    return out;
}

OptValPtr OptValue::tanh() const {
    double x = this->data;
    double func = (std::exp(2*x) - 1) / (std::exp(2*x) + 1);
    auto out = make_value(func);

    gradient_ops.push_back([self=std::const_pointer_cast<OptValue>(shared_from_this()), func, out]() {
            self->grad += (1 - func*func) * out->grad;
    });

    return out;
}

void OptValue::apply_gradients() {
    for (auto it = gradient_ops.rbegin(); it != gradient_ops.rend(); it++) {
        (*it)();
    }

    gradient_ops.clear();
}

