#include <iostream>
#include <vector>
#include <unordered_set>
#include <string>
#include <functional>
#include "tensorimpl.hpp"
#include "src\cpp\cpu\nn\optmizers\sgd.h"
using namespace mylib::tensor;
using namespace std;
void build_topo(TensorImpl* node, vector<TensorImpl*>& topo, unordered_set<TensorImpl*>& visited) {
    if (visited.count(node)) return;
    visited.insert(node);

    for (auto p : node->parents)
        build_topo(p, topo, visited);

    topo.push_back(node);
}
void optmizerengine(TensorImpl* loss)
 {

    vector<TensorImpl*> topo;
    unordered_set<TensorImpl*> visited;
    build_topo(loss, topo, visited);

    for (int i = topo.size() - 1; i >= 0; i--) {
        TensorImpl* node = topo[i];
        if (node->tooptmize==true){
            optmizerSGD(0.01,node);}
    }
}
void backward(TensorImpl* loss) {
    // 1. initialize loss grad
    float* input1Grad = static_cast<float*>(loss->storageforgrad->data);

    // 2. build topo order
    vector<TensorImpl*> topo;
    unordered_set<TensorImpl*> visited;
    build_topo(loss, topo, visited);

    // 3. walk backwards
   for (int i = topo.size() - 1; i >= 0; i--) {
        TensorImpl* node = topo[i];
        if (node->backwardfunction){
            node->backwardfunction(node);}
    }
}