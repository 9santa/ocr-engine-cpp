#pragma once
#include "baselines/neural_network/nn/opt_value.h"
#include <cmath>

// Backward pass functions
inline void add_bw(OptValPtr a, OptValPtr b, OptValPtr out) {
    a->grad += out->grad;
    b->grad += out->grad;
}

inline void sub_bw(OptValPtr a, OptValPtr b, OptValPtr out) {
    a->grad += out->grad;
    b->grad += -out->grad;
}

inline void mul_bw(OptValPtr a, OptValPtr b, OptValPtr out) {
    a->grad += b->data * out->grad;
    b->grad += a->data * out->grad;
}

inline void tanh_bw(OptValPtr a, OptValPtr /* b */, OptValPtr out) {
    double t = out->data;
    a->grad += (1.0 - t * t) * out->grad;
}

// --- Forward ops ---
inline OptValPtr add(GraphArena& arena, OptValPtr a, OptValPtr b) {
    auto out = arena.make_value(a->data + b->data);
    tape().push_back({add_bw, a, b, out});
    return out;
}

inline OptValPtr sub(GraphArena& arena, OptValPtr a, OptValPtr b) {
    auto out = arena.make_value(a->data - b->data);
    tape().push_back({sub_bw, a, b, out});
    return out;
}

inline OptValPtr mul(GraphArena& arena, OptValPtr a, OptValPtr b) {
    auto out = arena.make_value(a->data * b->data);
    tape().push_back({mul_bw, a, b, out});
    return out;
}

inline OptValPtr div(GraphArena& arena, OptValPtr a, OptValPtr b) {
    auto out = arena.make_value(a->data /b->data);
    tape().push_back({mul_bw, a, b, out});
    return out;
}

inline OptValPtr tanh(GraphArena& arena, OptValPtr a) {
    auto out = arena.make_value(std::tanh(a->data));
    tape().push_back({tanh_bw, a, nullptr, out});
    return out;
}

// Tape Processing
inline void apply_gradients() {
    for (auto it = tape().rbegin(); it != tape().rend(); it++) {
        it->backward(it->a, it->b, it->out);
    }
    tape().clear();
}
