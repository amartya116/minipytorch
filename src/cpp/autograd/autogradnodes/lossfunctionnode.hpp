#pragma once
#include "src\cpp\autograd\autogradnodes\lossfucntionnode.cpp"

TensorImpl* lossfunctionnode(TensorImpl* input1, const vector<float>& target,enum Dtypes dtype);
