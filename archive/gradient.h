#pragma once
#ifndef GRADIENT_H
#define GRADIENT_H

#include <memory>
#include <set>
#include <string>
#include <functional>

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

    Value(double data, std::set<ValPtr> _children = {}, std::string _op = "", double grad = 0.0, std::string label = "");

    // Operator overloads
    ValPtr operator+(const ValPtr& other);
    void operator+=(const ValPtr& other);
    ValPtr operator-(const ValPtr& other);
    ValPtr operator*(const ValPtr& other);
    ValPtr tanh();
    ValPtr relu();
    void backward();

    friend std::ostream& operator<<(std::ostream& os, const ValPtr& v);
};



#endif // !GRADIENT_H
