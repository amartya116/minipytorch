#include <iostream>
#include <immintrin.h>
#include "../../core/tensorimpl.hpp"
#include "../../cpu/nn/activation/relu.h"
#include "../../cpu/nn/lossfunction/mseloss.cpp"
using namespace mylib::tensor;

// Forward declaration
void MSElossfunctionnodebackward(TensorImpl* NodeInput, float* target);
TensorImpl* lossfunctionnode(TensorImpl* input1, float* target, enum mylib::core::Dtypes dtype)
{
    int resultdatasize = input1->storage->nbytes;
    auto resultStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
    auto resultGradStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
    TensorImpl *lossnodeMSE = new TensorImpl(resultStorage, resultGradStorage, input1->shape, input1->strides, 0,
        [target](TensorImpl* NodeInput){ MSElossfunctionnodebackward(NodeInput, target); }, "MSElossfunctionnodebackward", 0, true, {input1},false);
    float* input1Data = static_cast<float*>(input1->storage->data);
    float* resultData = static_cast<float*>(lossnodeMSE->storage->data);
    resultData=mseloss(input1Data, target, resultdatasize);
    return lossnodeMSE;
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

void BAElossfunctionnodebackward(TensorImpl* NodeInput, float* target){
    int i=0;
    TensorImpl* pred = NodeInput->parents[0];
    float* nodeGradData = static_cast<float*>(NodeInput->storageforgrad->data);
    float* nodeprevData = static_cast<float*>(pred->storage->data);
    int N = pred->storage->nbytes / sizeof(float);  // Convert bytes to float count
   // loss->grad[0] should be 1
    float dL_dloss = 1.0f;  // Initialize to 1 instead of reading uninitialized memory
    

    for (; i < N; i++) {
        float y_pred = nodeprevData[i];
        float y_true = target[i];
        float dloss_dpred = (1.0f / N) * (y_pred - y_true);  // d(MSE)/dpred
        nodeGradData[i] = dL_dloss * dloss_dpred;  // Initialize, not accumulate
    }
}