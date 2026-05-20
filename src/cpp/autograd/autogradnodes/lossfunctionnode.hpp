#pragma once
#include "src\cpp\autograd\autogradnodes\lossfunctionnode.cpp"

TensorImpl* lossfunctionnode(TensorImpl* input1, const vector<float>& target,enum Dtypes dtype);
