#include <iostream>
#include <immintrin.h>
#include "src\cpp\core\tensorimpl.hpp"
#include "src\cpp\cpu\nn\activation\relu.h"
using namespace std;
using namespace mylib::tensor;
TensorImpl* activationnode(TensorImpl* input1)
{
    int resultdatasize = input1->storage->nbytes;
    auto resultStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
    auto resultGradStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
    TensorImpl *activationnoderelu = new TensorImpl(resultStorage, resultGradStorage, input1->shape, input1->strides, 0, Relubackward, "Relubackward", 0, true, {input1});
}
void Relubackward(TensorImpl* NodeInput){
    int i=0;
    float* nodeGradData = static_cast<float*>(NodeInput->storageforgrad->data);
    for(;i<NodeInput->storage->nbytes;i=i++){
		nodeGradData[i]+=((nodeGradData[i] < 0) ? nodeGradData[i]*0 : nodeGradData[i]*1);
	}
}