#include <cmath>
#include <iomanip>
#include <ios>
#include <ostream>
#include <set>
#include <string>
#include <functional>
#include <iostream>

class Value {
public:
    double data;
    double grad;
    std::string label;
    std::string op;

    std::set<Value*> prev;
    std::function<void()> _backward;

    Value(double data, std::set<Value*> _children = {}, std::string _op = "", double grad = 0.0, std::string label = "")
        : data(data), grad(grad), label(label), op(_op), prev(_children)
    {
        _backward = []() {};
    }

    // Operator overloads
    Value operator+(const Value& other) {
        Value* a = this;
        Value* b = const_cast<Value*>(&other);
        Value out(a->data + b->data, {a, b}, "+");

        out._backward = [a, b, &out]() {
            a->grad = 1.0 * out.grad;
            b->grad = 1.0 * out.grad;
        };

        return out;
    }

    Value& operator+=(const Value& other) {
        *this = *this + other;
        return *this;
    }

    Value operator-(const Value& other) {
        Value* a = this;
        Value* b = const_cast<Value*>(&other);
        Value out(a->data - b->data, {a, b}, "-");

        return out;
    }

    Value operator*(const Value& other) {
        Value* a = this;
        Value* b = const_cast<Value*>(&other);
        Value out(a->data * b->data, {a, b}, op = "*");

        out._backward = [a, b, &out]() {
            a->grad = b->data * out.grad;
            b->grad = a->data * out.grad;
        };

        return out;
    }

    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        os << "Value " << v.label << "(data= " << v.data << ")\n";
        return os;
    }

    Value tanh() {
        double x = this->data;
        double func = (std::exp(2*x) - 1) / (std::exp(2*x) + 1);
        Value out(func, {this}, "tanh");

        auto _backward = [this, func, &out]() {
            this->grad += (1 - func*func) * out.grad;
        };

        out._backward = _backward;
        return out;
    }

    void backward() {
        std::vector<Value*> topo;
        std::set<Value*> visited;

        auto topologicalSort = [&](auto&& self, Value* v) -> void {
            if (!visited.contains(v)) {
                visited.insert(v);
                for (const auto& child : v->prev) {
                    self(self, child);
                }
                topo.push_back(v);
            }
        };

        topologicalSort(topologicalSort, this);
        this->grad = 1.0;

        for (auto it = topo.rbegin(); it != topo.rend(); it++) {
            (*it)->_backward();
        }
    }

    // activation function: rectified linear unit
    Value relu() {
        double relu_data = std::max(0.0, this->data);
        Value out(relu_data, {const_cast<Value*>(this)}, "ReLU");

        // define backward pass
        out._backward = [this, &out]() {
            this->grad += (this->data > 0 ? 1.0 : 0.0) * out.grad;
        };

        return out;
    }
};


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
