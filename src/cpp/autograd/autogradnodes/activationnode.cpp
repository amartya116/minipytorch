#include <iostream>
#include <immintrin.h>
#include "../../../core/tensorimpl.hpp"
#include "../../../cpu/nn/activation/relu.h"
using namespace std;
using namespace mylib::tensor;
TensorImpl* activationnode(TensorImpl* input1,enum Dtypes dtype)
{
    int resultdatasize = input1->storage->nbytes;
    auto resultStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
    auto resultGradStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
    TensorImpl *activationnoderelu = new TensorImpl(resultStorage, resultGradStorage, input1->shape, input1->strides, 0, Relubackward, "Relubackward", 0, true, {input1},false);
    return activationnoderelu;
}
//RELU node Autodiff
void Relubackward(TensorImpl* NodeInput){
    int i=0;
    float* nodeGradData = static_cast<float*>(NodeInput->storageforgrad->data);
    float* parent0GradData = static_cast<float*>(NodeInput->parents[0]->storageforgrad->data);
    float* parent0Data = static_cast<float*>(NodeInput->parents[0]->storage->data);  
   for(;i<NodeInput->storage->nbytes/sizeof(float);i=i++){
    parent0GradData[i] += (parent0Data[i] > 0) ? nodeGradData[i] : 0.0f;
}
}