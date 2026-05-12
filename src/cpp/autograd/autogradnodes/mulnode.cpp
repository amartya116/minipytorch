#include <iostream>
#include <vector>
#include <string>
#include <immintrin.h>
#include "src\cpp\core\tensorimpl.hpp"
#include "src\cpp\core\dtype.hpp"
#include "linear_algebra_matrix_operations/simdmatmul.hpp"
using namespace mylib::tensor;
using namespace  mylib::core;
using namespace std;

TensorImpl* matmul(TensorImpl *input1,TensorImpl *input2) {
	int resultdatasize = input1->storage->nbytes;
	int resultdatasize2 = input2->storage->nbytes;
	auto resultStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
	auto resultGradStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
		// Cast void* to float* for SIMD operations
	float* input1Data = static_cast<float*>(input1->storage->data);
	float* input2Data = static_cast<float*>(input2->storage->data);
	float* resultData = static_cast<float*>(resultStorage->data);
	matmulsimdgemmcacheaware(input1Data,input2Data, resultData, resultdatasize,resultdatasize2,result->size);
	return result;
}
void mulbackward(Tensornode* NodeInput){
	int i=0;
    for(;i<NodeInput->size-7;i=i+8){ 
		_mm256_storeu_ps(NodeInput->parents[0]->grad+i,_mm256_add_ps(_mm256_loadu_ps(NodeInput->parents[0]->grad+i),_mm256_mul_ps(_mm256_loadu_ps(NodeInput->parents[1]->data+i),_mm256_loadu_ps(NodeInput->grad+i))));
		_mm256_storeu_ps(NodeInput->parents[1]->grad+i,_mm256_add_ps(_mm256_loadu_ps(NodeInput->parents[1]->grad+i),_mm256_mul_ps(_mm256_loadu_ps(NodeInput->parents[0]->data+i),_mm256_loadu_ps(NodeInput->grad+i))));
    }
	for(;i<NodeInput->size;i=i++){
		NodeInput->parents[0]->grad[i]+=NodeInput->parents[1]->data[i]*NodeInput->grad[i];
        NodeInput->parents[1]->grad[i]+=NodeInput->parents[0]->data[i]*NodeInput->grad[i];
	}
}