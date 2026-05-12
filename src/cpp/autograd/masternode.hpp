#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include "tensorimpl.hpp"
using namespace std;
using namespace mylib::tensor;
std::shared_ptr<TensorImpl> impl;
namespace mylib::autograd{
struct Tensornode{
    float* data;
    float* grad;
    float singlevaluedata;
    vector<int> shape;
    int size;
    bool tooptmize;
    string device;
    string backwardfnname;
    function<void(Tensornode*)> backwardfunction;
    int level;
    bool requiregrad;
    vector<Tensornode*> parents;
};
struct autogradgraph
{
    vector<float> accumaltedgrad;
    function<void()> function1;
    function<void()> function2;
};}