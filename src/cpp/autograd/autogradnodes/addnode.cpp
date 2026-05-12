#include <iostream>
#include <vector>
#include <string>
#include "src\cpp\core\tensorimpl.hpp"
#include "src\cpp\core\dtype.hpp"

#include "linear_algebra_matrix_operations/simdadd.hpp"
using namespace std;
using namespace mylib::tensor;
using namespace  mylib::core;
TensorImpl* add(TensorImpl *input1,TensorImpl *input2) {
	int resultdatasize = input1->storage->nbytes;
	auto resultStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
	auto resultGradStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
	
	// Cast void* to float* for SIMD operations
	float* input1Data = static_cast<float*>(input1->storage->data);
	float* input2Data = static_cast<float*>(input2->storage->data);
	float* resultData = static_cast<float*>(resultStorage->data);
	
	simdadd(input1Data, input2Data, resultData, 1, resultdatasize);
	TensorImpl *result = new TensorImpl(resultStorage, resultGradStorage, input1->shape, input1->strides, 0, addbackward, "addbackward", 0, true, {input1, input2});
	
	return result;
}
void addbackward(TensorImpl* NodeInput){
	int i=0;
	float* nodeGradData = static_cast<float*>(NodeInput->storageforgrad->data);
	
	for(;i<NodeInput->storage->nbytes;i=i++){
        float* parent0GradData = static_cast<float*>(NodeInput->parents[0]->storageforgrad->data);
        float* parent1GradData = static_cast<float*>(NodeInput->parents[1]->storageforgrad->data);
        parent0GradData[i]+=nodeGradData[i];
        parent1GradData[i]+=nodeGradData[i];
    }
}