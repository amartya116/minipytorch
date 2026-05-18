#pragma once
#include "src\cpp\autograd\autogradnodes\mulnode.cpp"

TensorImpl* matmul(TensorImpl *input1, TensorImpl *input2,enum Dtypes dtype);