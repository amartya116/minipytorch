#include <iostream>
#include <vector>
#include <string>
#include <immintrin.h>
#include "../masternode.hpp"
#include "linear_algebra_matrix_operations/simdmatmul.hpp"

using namespace std;
using namespace mylib::autograd;

Tensornode* matmul(Tensornode *input1,Tensornode *input2) {
	Tensornode *result = new Tensornode();
	int resultdatasize = input1->size;
	result->size = resultdatasize;
	result->shape = input1->shape;
	result->data = new float[resultdatasize];
	matmulsimdgemmcacheaware(input1->data, input2->data, result->data, input1->size, input2->size,result->size);
	result->backwardfnname = "mulbackward";
    result->backwardfunction=mulbackward;
	result->parents.push_back(input1);
	result->parents.push_back(input2);
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