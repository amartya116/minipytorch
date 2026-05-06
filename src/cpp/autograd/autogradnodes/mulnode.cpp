#include <iostream>
#include <vector>
#include <string>
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
    for(int i=0;i<NodeInput->size;i++){
        NodeInput->parents[0]->grad[i]+=NodeInput->parents[1]->data[i]*NodeInput->grad[i];
        NodeInput->parents[1]->grad[i]+=NodeInput->parents[0]->data[i]*NodeInput->grad[i];
    }
}