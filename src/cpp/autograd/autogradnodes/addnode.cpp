#include <iostream>
#include <vector>
#include <string>
#include <immintrin.h>
#include "../../core/tensorimpl.hpp"
#include "../../core/dtype.hpp"

#include "../../linear_algebra_matrix_operations/simdadd.hpp"
using namespace std;
using namespace mylib::tensor;
using namespace  mylib::core;
void addbackward(TensorImpl* NodeInput);
TensorImpl* add(TensorImpl *input1,TensorImpl *input2,Dtypes dtype) {
	int resultdatasize = input1->storage->nbytes; 
	auto resultStorage = make_shared<Storage>(resultdatasize, dtype, input1->storage->device);
	auto resultGradStorage = make_shared<Storage>(resultdatasize, dtype, input1->storage->device);
	
	// Cast void* to float* for SIMD operations
	float* input1Data = static_cast<float*>(input1->storage->data);
	float* input2Data = static_cast<float*>(input2->storage->data);
	float* resultData = static_cast<float*>(resultStorage->data);
	
	simdadd(input1Data, input2Data, resultData, 1, resultdatasize/sizeof(float));
	TensorImpl *result = new TensorImpl(resultStorage, resultGradStorage, input1->shape, input1->strides, 0, addbackward, "addbackward", 0, true, {input1, input2}, false);
	
	return result;
}
// bias add: input {rows, cols} + bias {cols}, bias broadcast across rows
void addbiasbackward(TensorImpl* NodeInput);
TensorImpl* addbias(TensorImpl *input, TensorImpl *bias, Dtypes dtype) {
	int rows = input->shape[0];
	int cols = input->shape[1];
	int resultdatasize = input->storage->nbytes;
	auto resultStorage = make_shared<Storage>(resultdatasize, dtype, input->storage->device);
	auto resultGradStorage = make_shared<Storage>(resultdatasize, dtype, input->storage->device);

	float* inputData = static_cast<float*>(input->storage->data);
	float* biasData = static_cast<float*>(bias->storage->data);
	float* resultData = static_cast<float*>(resultStorage->data);

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			resultData[r * cols + c] = inputData[r * cols + c] + biasData[c];
		}
	}
	return new TensorImpl(resultStorage, resultGradStorage, input->shape, input->strides, 0, addbiasbackward, "addbiasbackward", 0, true, {input, bias}, false);
}
void addbiasbackward(TensorImpl* NodeInput) {
	int rows = NodeInput->shape[0];
	int cols = NodeInput->shape[1];
	float* nodeGradData = static_cast<float*>(NodeInput->storageforgrad->data);
	float* parent0GradData = static_cast<float*>(NodeInput->parents[0]->storageforgrad->data);
	float* parent1GradData = static_cast<float*>(NodeInput->parents[1]->storageforgrad->data);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			parent0GradData[r * cols + c] += nodeGradData[r * cols + c];
			parent1GradData[c] += nodeGradData[r * cols + c];
		}
	}
}
//add node Autodiff
void addbackward(TensorImpl* NodeInput){
	int i=0;
	float* nodeGradData = static_cast<float*>(NodeInput->storageforgrad->data);
	float* parent0GradData = static_cast<float*>(NodeInput->parents[0]->storageforgrad->data);
    float* parent1GradData = static_cast<float*>(NodeInput->parents[1]->storageforgrad->data);
   for(;i+7<(NodeInput->storage->nbytes/sizeof(float));i=i+8){ 
		_mm256_storeu_ps(parent0GradData+i,_mm256_add_ps(_mm256_loadu_ps(parent0GradData+i),_mm256_loadu_ps(nodeGradData+i)));
		_mm256_storeu_ps(parent1GradData+i,_mm256_add_ps(_mm256_loadu_ps(parent1GradData+i),_mm256_loadu_ps(nodeGradData+i)));
    }
	for(;i<NodeInput->storage->nbytes/sizeof(float);i++){
		parent0GradData[i]+=nodeGradData[i];
        parent1GradData[i]+=nodeGradData[i];
	}
} 