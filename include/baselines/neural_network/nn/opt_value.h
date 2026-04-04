#pragma once
#include <vector>

// Forward declarations
struct OptValue;
struct GraphArena;
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
OptValPtr add(GraphArena& arena, OptValPtr a, OptValPtr b);
OptValPtr sub(GraphArena& arena, OptValPtr a, OptValPtr b);
OptValPtr mul(GraphArena& arena, OptValPtr a, OptValPtr b);
OptValPtr div(GraphArena& arena, OptValPtr a, OptValPtr b);
OptValPtr tanh(GraphArena& arena, OptValPtr a);
void apply_gradients();

struct OptValue {
    double data;
    double grad;

    explicit OptValue(double _data) : data(_data), grad(0.0) {}

    inline void zero_grad() { grad = 0.0; }

    // for parameters (weights, biases) - long lived
    inline static OptValPtr make_param(double x) {
        return new OptValue(x);
    }
};

// Arena for per-step temporaries
struct GraphArena {
    std::vector<OptValPtr> nodes;

    explicit GraphArena(size_t reserve_count = 0) {
        nodes.reserve(reserve_count);
    }

    // for activations / intermediates - short lived, owned by the arena
    OptValPtr make_value(double x) {
        OptValPtr p = new OptValue(x);
        nodes.push_back(p);
        return p;
    }

    void clear() {
        for (auto* p : nodes) {
            delete p;
        }
        nodes.clear();
    }

    ~GraphArena() {
        clear();
    }
};
