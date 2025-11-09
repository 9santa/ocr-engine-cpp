#include "gradient.cpp"
#include <cstddef>
#include <memory>
#include <random>
#include <variant>


class Neuron {
public:
    std::vector<ValPtr> w;   // weights
    ValPtr b;                // bias
    bool nonlin;

    Neuron(int nin, bool nonlin=true) : nonlin(nonlin) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        for (int i = 0; i < nin; i++) {
            w.push_back(std::make_shared<Value>(dis(gen)));
        }
        b = std::make_shared<Value>(0.0);
    }

    ValPtr operator()(const std::vector<ValPtr>& x) {
        ValPtr act = b;
        for (size_t i = 0; i < w.size(); i++) {
            ValPtr wx = *w[i] * x[i];
            act = *act + wx;
        }
        // return nonlin ? act.relu() : act;
        return nonlin ? act->tanh() : act;
    }

    std::vector<ValPtr> parameters() {
        std::vector<ValPtr> params = w;
        params.push_back(b);
        return params;
    }
};

class Layer {
public:
    std::vector<Neuron> neurons;
    int nin, nout;

    Layer(int nin, int nout) : nin(nin), nout(nout) {
        for (int i = 0; i < nout; i++) {
            neurons.emplace_back(nin);
        }
    }

    using LayerOutput = std::variant<Value, std::vector<Value>>;
    std::vector<ValPtr> operator()(const std::vector<ValPtr>& x) {
        std::vector<ValPtr> outs;
        for (auto& n : neurons) outs.push_back(n(x));

        // return (outs.size() == 1 ? outs[0] : outs);
        return outs;
    }

    std::vector<ValPtr> parameters() {
        std::vector<ValPtr> params;
        for (auto& n : neurons) {
            auto np = n.parameters();
            params.insert(params.end(), np.begin(), np.end());
        }
        return params;
    }
};

class MLP {
public:
    std::vector<Layer> layers;

public:
    MLP(int nin, const std::vector<int>& nouts) {
        std::vector<int> sz = {nin};
        sz.insert(sz.end(), nouts.begin(), nouts.end());

        for (size_t i = 0; i < nouts.size(); i++) {
            layers.emplace_back(sz[i], sz[i+1]);
        }

    }

    std::vector<ValPtr> operator()(const std::vector<ValPtr>& x) {
        auto output = x;
        for (auto& layer : layers) {
            output = layer(output);
        }
        return output;
    }

    std::vector<ValPtr> parameters() {
        std::vector<ValPtr> params;
        for (auto& l : layers) {
            std::vector<ValPtr> lp = l.parameters();
            params.insert(params.end(), lp.begin(), lp.end());
        }
        return params;
    }
};

// loss function: Mean-Squared Error
ValPtr MSE(const std::vector<ValPtr>& ys, const std::vector<ValPtr>& ypred) {
    ValPtr total_loss = std::make_shared<Value>(0.0);
    for (size_t i = 0; i < ys.size(); i++) {
        auto diff = *ys[i] - ypred[i];
        auto sq = *diff * diff;
        total_loss = *total_loss + sq;
    }
    return total_loss;
}

void train(MLP& l, std::vector<std::vector<ValPtr>> input, const std::vector<ValPtr>& ys) {
    std::vector<ValPtr> final_pred;
    int iterations = 40;
    for (int i = 0; i <= iterations; i++) {
        // zero gradients
        for (auto& p : l.parameters()) {
            p->grad = 0.0;
        }

        // forward pass
        std::vector<ValPtr> ypred;
        for (auto&x : input) {
            ypred.push_back(l(x)[0]);
        }

        auto loss = MSE(ys, ypred);

        // backward pass
        loss->backward();

        // update (gradient descent)
        for (auto& p : l.parameters()) {
            p->data -= 0.1 * p->grad;
        }

        std::cout << "Iteration: " << i << ", loss=" << loss->data << "\n";

        if (i == iterations) {
            final_pred.insert(final_pred.end(), ypred.begin(), ypred.end());
        }
    }

    for (auto pred : final_pred) std::cout << pred;
}


signed main(void) {
    #if 0
    std::vector<Value> x = {2.0, 3.0, -1.0};
    MLP l(3, {4, 4, 1});
    auto output = l(x);

    for (auto& out : output) std::cout << out;
    #endif

    std::vector<std::vector<ValPtr>> xs = {
        {std::make_shared<Value>(2.0), std::make_shared<Value>(3.0), std::make_shared<Value>(-1.0)},
        {std::make_shared<Value>(3.0), std::make_shared<Value>(-1.0), std::make_shared<Value>(0.5)},
        {std::make_shared<Value>(0.5), std::make_shared<Value>(1.0), std::make_shared<Value>(1.0)},
        {std::make_shared<Value>(1.0), std::make_shared<Value>(1.0), std::make_shared<Value>(-1.0)}
    };

    std::vector<ValPtr> ys = {
        std::make_shared<Value>(1.0),
        std::make_shared<Value>(-1.0),
        std::make_shared<Value>(-1.0),
        std::make_shared<Value>(1.0)
    };

    MLP l(3, {4, 4, 1});

    train(l, xs, ys);


    // std::vector<Value> ypred;
    // for (auto& x : xs) {
    //     ypred.push_back(l(x)[0]);
    // }

    // for (const auto& pred : ypred) std::cout << pred;

    // auto loss = MSE(ys, ypred);
    // for (const auto& l : loss) std::cout << l;
    // total loss
    // std::cout << "Total loss: " << loss[0];

    // std::cout << l.layers[0].neurons[0].w[0].grad;

    // for (const auto& param : l.parameters()) {
    //     std::cout << param;
    // }

    return 0;
}
