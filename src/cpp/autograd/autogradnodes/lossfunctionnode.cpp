#include <iostream>
#include <algorithm>
#include <immintrin.h>
#include "../../core/tensorimpl.hpp"
#include "../../cpu/nn/activation/relu.h"
#include "../../cpu/nn/lossfunction/mseloss.cpp"
#include "../../cpu/nn/lossfunction/maeloss.cpp"
#include "../../cpu/nn/lossfunction/binarycrosslossentropy.cpp"
using namespace mylib::tensor;

// Forward declaration
void MSElossfunctionnodebackward(TensorImpl* NodeInput, float* target);
void MAElossfunctionnodebackward(TensorImpl* NodeInput, float* target);
void BCElossfunctionnodebackward(TensorImpl* NodeInput, float* target);
TensorImpl* lossfunctionnode(TensorImpl* input1, float* target, enum mylib::core::Dtypes dtype)
{
    int resultdatasize = input1->storage->nbytes;
    auto resultStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
    auto resultGradStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
    TensorImpl *lossnodeMSE = new TensorImpl(resultStorage, resultGradStorage, input1->shape, input1->strides, 0,
        [target](TensorImpl* NodeInput){ MSElossfunctionnodebackward(NodeInput, target); }, "MSElossfunctionnodebackward", 0, true, {input1},false);
    tape.push_back(lossnodeMSE);
    float* input1Data = static_cast<float*>(input1->storage->data);
    float* resultData = static_cast<float*>(lossnodeMSE->storage->data);
    resultData[0]=mseloss(input1Data, target, resultdatasize/sizeof(float));
    return lossnodeMSE;
}
//MAE node forward
TensorImpl* MAElossfunctionnode(TensorImpl* input1, float* target, enum mylib::core::Dtypes dtype)
{
    int resultdatasize = input1->storage->nbytes;
    auto resultStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
    auto resultGradStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
    TensorImpl *lossnodeMAE = new TensorImpl(resultStorage, resultGradStorage, input1->shape, input1->strides, 0,
        [target](TensorImpl* NodeInput){ MAElossfunctionnodebackward(NodeInput, target); }, "MAElossfunctionnodebackward", 0, true, {input1},false);
    tape.push_back(lossnodeMAE);
    float* input1Data = static_cast<float*>(input1->storage->data);
    float* resultData = static_cast<float*>(lossnodeMAE->storage->data);
    resultData[0]=MAEloss(input1Data, target, resultdatasize/sizeof(float));
    return lossnodeMAE;
}
//BCE node forward
TensorImpl* BCElossfunctionnode(TensorImpl* input1, float* target, enum mylib::core::Dtypes dtype)
{
    int resultdatasize = input1->storage->nbytes;
    auto resultStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
    auto resultGradStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
    TensorImpl *lossnodeBCE = new TensorImpl(resultStorage, resultGradStorage, input1->shape, input1->strides, 0,
        [target](TensorImpl* NodeInput){ BCElossfunctionnodebackward(NodeInput, target); }, "BCElossfunctionnodebackward", 0, true, {input1},false);
    tape.push_back(lossnodeBCE);
    float* input1Data = static_cast<float*>(input1->storage->data);
    float* resultData = static_cast<float*>(lossnodeBCE->storage->data);
    resultData[0]=BCEloss(input1Data, target, resultdatasize/sizeof(float));
    return lossnodeBCE;
}
//MSE node Autodiff
void MSElossfunctionnodebackward(TensorImpl* NodeInput, float* target){
    int i=0;
    TensorImpl* pred = NodeInput->parents[0];
    float* nodeGradData = static_cast<float*>(pred->storageforgrad->data);
    float* nodeprevData = static_cast<float*>(pred->storage->data);
    int N = pred->storage->nbytes / sizeof(float);  // Convert bytes to float count
   // loss->grad[0] should be 1
    float dL_dloss = 1.0f;  // Initialize to 1 instead of reading uninitialized memory


    for (; i < N; i++) {
        float y_pred = nodeprevData[i];
        float y_true = target[i];
        float dloss_dpred = (2.0f / N) * (y_pred - y_true);  // d(MSE)/dpred
        nodeGradData[i] = dL_dloss * dloss_dpred;  // Initialize, not accumulate
    }
}
template <typename T>
    int sign(T val) {
        return (T(0) < val) - (val < T(0));
}
void MAElossfunctionnodebackward(TensorImpl* NodeInput, float* target){
    int i=0;
    TensorImpl* pred = NodeInput->parents[0];
    float* nodeGradData = static_cast<float*>(pred->storageforgrad->data);
    float* nodeprevData = static_cast<float*>(pred->storage->data);
    int N = pred->storage->nbytes / sizeof(float);  // Convert bytes to float count
   // loss->grad[0] should be 1
    float dL_dloss = 1.0f;  // Initialize to 1 instead of reading uninitialized memory


    for (; i < N; i++) {
        float y_pred = nodeprevData[i];
        float y_true = target[i];
        float dloss_dpred = (1.0f / N) * sign((y_pred - y_true));  // d(MSE)/dpred
        nodeGradData[i] = dL_dloss * dloss_dpred;  // Initialize, not accumulate
    }
}

//BCE node Autodiff
void BCElossfunctionnodebackward(TensorImpl* NodeInput, float* target){
    int i=0;
    TensorImpl* pred = NodeInput->parents[0];
    float* nodeGradData = static_cast<float*>(pred->storageforgrad->data);
    float* nodeprevData = static_cast<float*>(pred->storage->data);
    int N = pred->storage->nbytes / sizeof(float);  // Convert bytes to float count
   // loss->grad[0] should be 1
    float dL_dloss = 1.0f;  // Initialize to 1 instead of reading uninitialized memory
    float eps = 1e-7f;      // keep p away from 0 and 1 so p*(1-p) never hits 0


    for (; i < N; i++) {
        float y_pred = std::max(eps, std::min(nodeprevData[i], 1.0f - eps));
        float y_true = target[i];
        float dloss_dpred = (1.0f / N) * (y_pred - y_true) / (y_pred * (1.0f - y_pred));  // d(BCE)/dpred
        nodeGradData[i] = dL_dloss * dloss_dpred;  // Initialize, not accumulate
    }
}
