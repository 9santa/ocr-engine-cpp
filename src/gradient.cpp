#include <cmath>
#include <ctime>
#include <iomanip>
#include <ios>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <functional>
#include <iostream>

class Value;
using ValPtr = std::shared_ptr<Value>;

class Value : public std::enable_shared_from_this<Value> {
public:
    double data;
    double grad;
    std::string label;
    std::string op;

    std::set<ValPtr> prev;
    std::function<void()> _backward;

    Value(double data, std::set<ValPtr> _children = {}, std::string _op = "", double grad = 0.0, std::string label = "")
        : data(data), grad(grad), label(label), op(_op), prev(_children)
    {
        _backward = []() {};
    }

    // Operator overloads
    ValPtr operator+(const ValPtr& other) {
        ValPtr out = std::make_shared<Value>(data + other->data, std::set<ValPtr>{shared_from_this(), other}, "+");

        out->_backward = [self=shared_from_this(), other, out]() {
            self->grad += 1.0 * out->grad;
            other->grad += 1.0 * out->grad;
        };

        return out;
    }

    void operator+=(const ValPtr& other) {
        ValPtr tmp = *shared_from_this() + other;

        data = tmp->data;
        grad = tmp->grad;
        op = tmp->op;
        prev = tmp->prev;
        _backward = tmp->_backward;
    }

    ValPtr operator-(const ValPtr& other) {
        ValPtr out = std::make_shared<Value>(data - other->data, std::set<ValPtr>{shared_from_this(), other}, "-");

        out->_backward = [self=shared_from_this(), other, out]() {
            self->grad += 1.0 * out->grad;
            other->grad += -1.0 * out->grad;
        };

        return out;
    }

    ValPtr operator*(const ValPtr& other) {
        ValPtr out = std::make_shared<Value>(data * other->data, std::set<ValPtr>{shared_from_this(), other}, "*");

        out->_backward = [self=shared_from_this(), other, out]() {
            self->grad += other->data * out->grad;
            other->grad += self->data * out->grad;
        };

        return out;
    }

    friend std::ostream& operator<<(std::ostream& os, const ValPtr& v) {
        os << "Value " << v->label << "(data= " << v->data << ")\n";
        return os;
    }

    ValPtr tanh() {
        double x = this->data;
        double func = (std::exp(2*x) - 1) / (std::exp(2*x) + 1);
        ValPtr out = std::make_shared<Value>(func, std::set<ValPtr>{shared_from_this()}, "tanh");

        out->_backward = [self=shared_from_this(), func, out]() {
            self->grad += (1 - func*func) * out->grad;
        };

        return out;
    }

    void backward() {
        std::vector<ValPtr> topo;
        std::set<ValPtr> visited;

        auto topologicalSort = [&](auto&& self, ValPtr v) -> void {
            if (!visited.contains(v)) {
                visited.insert(v);
                for (const auto& child : v->prev) {
                    self(self, child);
                }
                topo.push_back(v);
            }
        };

        topologicalSort(topologicalSort, shared_from_this());
        this->grad = 1.0;

        for (auto it = topo.rbegin(); it != topo.rend(); it++) {
            (*it)->_backward();
        }
    }

    // activation function: rectified linear unit
    ValPtr relu() {
        double relu_data = std::max(0.0, this->data);
        ValPtr out = std::make_shared<Value>(relu_data, std::set<ValPtr>{shared_from_this()}, "ReLU");

        // define backward pass
        out->_backward = [self=shared_from_this(), out]() {
            self->grad += (self->data > 0 ? 1.0 : 0.0) * out->grad;
        };

        return out;
    }
};

#if 0
signed main1(void) {
    double delta = 0.0001;

    Value a(2.0, {}, "", 0.0, "a");
    Value b(-3.0); b.label = "b";
    Value c(10.0); c.label = "c";
    auto e = a*b; e.label = "e";
    auto d = e+c; d.label = "d";
    Value f(-2.0); f.label = "f";
    auto L1 = d * f; L1.label = "L1";

    auto L = L1;

    // derivative with respect to a
    a.data += delta;
    e = a*b; e.label = "e";
    d = e+c; d.label = "d";
    auto L2 = d * f; L2.label = "L2";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "dL/da = " << (L2.data - L1.data)/delta << "\n";


    // derivative with respect to itself is obv = 1
    L.grad = 1.0;

    // dL/dd = f
    // dL/df = d

    /* dL/dc = ?
       dd/dc = 1.0
       dd/de = 1.0

       dL/dc = dL/dd * dd/dc = f * 1.0 = f


    */

    std::cout << a;




    return 0;
}
#endif
