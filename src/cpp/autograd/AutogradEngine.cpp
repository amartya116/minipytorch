#include <iostream>
#include <vector>
#include <unordered_set>
#include <string>
#include <cstring>
#include <functional>
#include "../core/tensorimpl.hpp"
#include "../cpu/nn/optmizers/sgd.h"
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
// zero every grad buffer reachable from loss (leaves keep their grads across steps, so they must be cleared by hand)
void zerograd(TensorImpl* loss) {
    vector<TensorImpl*> topo;
    unordered_set<TensorImpl*> visited;
    build_topo(loss, topo, visited);
    for (int i = 0; i < topo.size(); i++) {
        TensorImpl* node = topo[i];
        memset(node->storageforgrad->data, 0, node->storageforgrad->nbytes);
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
            node->backwardfunction(node);
        }
    }
}