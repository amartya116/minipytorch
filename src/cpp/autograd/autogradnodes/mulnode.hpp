#pragma once

#include "../../../core/tensorimpl.hpp"
#include "../../../core/dtype.hpp"

using namespace mylib::tensor;
namespace mylib::autograd {
    TensorImpl* matmul(TensorImpl *input1, TensorImpl *input2, enum Dtypes dtype);
}

