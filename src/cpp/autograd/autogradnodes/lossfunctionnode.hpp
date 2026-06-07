#pragma once

#include <vector>

#include "../../../core/tensorimpl.hpp"
#include "../../../core/dtype.hpp"

using namespace mylib::tensor;
namespace mylib::autograd {
    TensorImpl* lossfunctionnode(TensorImpl* input1, const std::vector<float>& target, enum Dtypes dtype);
}

