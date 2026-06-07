#pragma once

#include "../../../core/tensorimpl.hpp"
#include "../../../core/dtype.hpp"

using namespace mylib::tensor;
namespace mylib::autograd {
    TensorImpl* activationnode(TensorImpl* input1, enum Dtypes dtype);
}

