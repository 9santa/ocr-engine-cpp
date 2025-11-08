#include "gradient.cpp"
#include <cstddef>
#include <random>
#include <variant>


class Neuron {
public:
    std::vector<Value> w;   // weights
    Value b;                // bias
    bool nonlin;

    Neuron(int nin, bool nonlin=true) 
        : b(0.0), nonlin(nonlin)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        for (int i = 0; i < nin; i++) {
            w.push_back(Value(dis(gen)));
        }
    }

    Value operator()(const std::vector<Value>& x) {
        Value act = b;
        for (size_t i = 0; i < w.size(); i++) {
            act += w[i] * x[i];
        }
        // return nonlin ? act.relu() : act;
        return act.tanh();
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
    std::vector<Value> operator()(const std::vector<Value>& x) {
        std::vector<Value> outs;
        for (auto& n : neurons) {
            outs.push_back(n(x));
        }
        // return (outs.size() == 1 ? outs[0] : outs);
        return outs;
    }
};

class MLP {
private:
    std::vector<Layer> layers;

public:
    MLP(int nin, const std::vector<int>& nouts) {
        std::vector<int> sz = {nin};
        sz.insert(sz.end(), nouts.begin(), nouts.end());

        for (size_t i = 0; i < nouts.size(); i++) {
            layers.emplace_back(sz[i], sz[i+1]);
        }

    }

    std::vector<Value> operator()(const std::vector<Value>& x) {
        auto output = x;
        for (auto& layer : layers) {
            output = layer(output);
        }
        return output;
    }
};

// loss function: Mean-Squared Error
std::vector<Value> MSE(std::vector<Value>& ys, std::vector<Value>& ypred) {
    std::vector<Value> losses;
    Value total_loss(0.0);
    for (size_t i = 0; i < ys.size(); i++) {
        auto diff = ys[i] - ypred[i];
        auto diffsq = diff*diff;
        losses.push_back(diffsq);
        total_loss += diffsq;
    }

    losses.insert(losses.begin(), total_loss);
    return losses;
}



signed main(void) {
    #if 0
    std::vector<Value> x = {2.0, 3.0, -1.0};
    MLP l(3, {4, 4, 1});
    auto output = l(x);

    for (auto& out : output) std::cout << out;
    #endif

    std::vector<std::vector<Value>> xs = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };
    std::vector<Value> ys = {1.0, -1.0, -1.0, 1.0};

    MLP l(3, {4, 4, 1});
    std::vector<Value> ypred;
    for (auto& x : xs) {
        ypred.push_back(l(x)[0]);
    }

    // for (const auto& pred : ypred) std::cout << pred;

    auto loss = MSE(ys, ypred);
    for (const auto& l : loss) std::cout << l;
    // total loss
    std::cout << "Total loss: " << loss[0];

    return 0;
}
