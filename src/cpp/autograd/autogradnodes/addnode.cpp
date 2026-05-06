#include <iostream>
#include <vector>
#include <string>
#include "../masternode.hpp"
#include "linear_algebra_matrix_operations/simdadd.hpp"
using namespace std;
using namespace mylib::autograd;
Tensornode* add(Tensornode *input1,Tensornode *input2) {
	Tensornode *result = new Tensornode();
	int resultdatasize = input1->size;
	result->size = resultdatasize;
	result->shape = input1->shape;
	result->data = new float[resultdatasize];
	simdadd(input1->data, input2->data, result->data, 1, resultdatasize);
	result->backwardfnname = "addbackward";
    result->backwardfunction=addbackward;
	result->parents.push_back(input1);
	result->parents.push_back(input2);
	return result;
}
void addbackward(Tensornode* NodeInput){
    for(int i=0;i<NodeInput->size;i++){
        NodeInput->parents[0]->grad[i]+=NodeInput->grad[i];
        NodeInput->parents[1]->grad[i]+=NodeInput->grad[i];
    }
}