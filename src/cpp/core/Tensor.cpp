#include <iostream>
#include <memory>
#include <vector>
#include "tensorimpl.hpp"

namespace mylib::tensor
{
struct Tensor
{
	std::shared_ptr<TensorImpl> impl;
	Tensor(shared_ptr<TensorImpl> impl) {
		this->impl = impl;
	}
	vector<int> shape() {
		return impl->getshape();
	}
	vector<int> strides() {
		return impl->getstrides();
	}
	int numel() {
		return impl->numel();
	}
	mylib::core::Dtypes dtype() {
		return impl->dtype;
	}
	mylib::core::device device() {
		return impl->device;
	}
	void* dataptr() {
		return impl->storage->dataptr();
	}
	bool iscontiguous() {
		return impl->iscontiguous();
	}



};
}