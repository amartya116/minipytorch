#include <iostream>
#include "dtype.hpp"
#include "device.hpp"
#include "allocator.hpp"

namespace mylib::tensor {
struct Storage
{
	void* data;
	size_t  nbytes;
	mylib::core::Dtypes dtype;
	mylib::core::device device;
	mylib::core::CPUAllocator allocator;

	Storage(size_t nbytes,mylib::core::Dtypes dtype,mylib::core::device device)
	{
		data=allocator.allocate(nbytes);
		this->nbytes = nbytes;
		this->dtype = dtype;
		this->device = device;
	}
	~Storage()
	{
		allocator.free(data, nbytes);
	}
	void* dataptr() {
		return data;
	}
	size_t size() {
		return nbytes;
	}


};
};