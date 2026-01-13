#pragma once
#include <vector>

struct OptValue;
using OptValPtr = OptValue*;
using BackwardFn = void(*)(OptValPtr, OptValPtr, OptValPtr);

struct TapeOp {
    BackwardFn backward;
    OptValPtr a;
    OptValPtr b;
    OptValPtr out;
};

inline std::vector<TapeOp>& tape() {
    static std::vector<TapeOp> t;
    return t;
}

// Forward declaration of ops
OptValPtr add(OptValPtr a, OptValPtr b);
OptValPtr mul(OptValPtr a, OptValPtr b);
OptValPtr tanh(OptValPtr a);
void apply_gradients();

struct OptValue {
    double data;
    double grad;

    explicit OptValue(double _data) : data(_data), grad(0.0) {}

    inline void zero_grad() { grad = 0.0; }

    inline static OptValPtr make_value(double x) {
        return new OptValue(x);
    }
};
