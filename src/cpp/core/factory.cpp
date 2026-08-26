#include <iostream>
#include "tensorimpl.hpp"
#include <vector>
#include <atomic>
#include <memory>
namespace mylib::tensor {

shared_ptr<TensorImpl> empty(vector<int> shape,
                             mylib::core::Dtypes dtype,
                             mylib::core::device dev) {
    int numel = 1;
    for (int s : shape) numel *= s;
    int nbytes = numel * mylib::core::itemsize(dtype);
    auto storage = make_shared<Storage>(nbytes, dtype, dev);
    auto storageforgrad = make_shared<Storage>(nbytes, dtype, dev);
    vector<int> strides(shape.size());
    auto noBackward = [](TensorImpl*) {};
    strides.back() = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    return make_shared<TensorImpl>(storage,storageforgrad, shape, strides, 0,noBackward, "null", 0, true, {},false);
}

shared_ptr<TensorImpl> zero(vector<int> shape,
                            mylib::core::Dtypes dtype,
                            mylib::core::device dev) {
    auto result = empty(shape, dtype, dev);

    int numel = 1;
    for (int s : shape) numel *= s;

    int nbytes = numel * mylib::core::itemsize(dtype);
    memset(result->storage->dataptr(), 0, nbytes);
    memset(result->storageforgrad->dataptr(), 0, nbytes);

    return result;
}

shared_ptr<TensorImpl> ones(vector<int> shape,
                            mylib::core::Dtypes dtype,
                            mylib::core::device dev) {
    auto result = empty(shape, dtype, dev);

    int numel = 1;
    for (int s : shape) numel *= s;

    float* data = static_cast<float*>(result->storage->dataptr());

    for (int i = 0; i < numel; i++) {
        data[i] = 1.0f;
    }

    return result;
}

shared_ptr<Storage> createbufferforopt(int size, mylib::core::Dtypes dtype, mylib::core::device dev) {
    int nbytes = size * mylib::core::itemsize(dtype);
    auto stateStorage = make_shared<Storage>(nbytes, dtype, dev);
    memset(stateStorage->dataptr(), 0, nbytes);
    return stateStorage;
}

}
