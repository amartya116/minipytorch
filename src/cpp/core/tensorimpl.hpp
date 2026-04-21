#ifndef TENSORIMPL_HPP
#define TENSORIMPL_HPP

#include <cstddef>
#include <memory>
#include <vector>
#include "storage.hpp"

namespace mylib::tensor
{
struct TensorImpl
{
	shared_ptr<Storage> storage;
	vector<int> shape;
	vector<int> strides;
	int offset;
	mylib::core::Dtypes dtype;
	mylib::core::device device;
	TensorImpl(shared_ptr<Storage> storage, vector<int> shape, vector<int> strides, int offset){
	    this->storage=storage;
	    this->shape=shape;
	    this->strides=strides;
	    this->offset=offset;
	    this->dtype=storage->dtype;
	    this->device=storage->device;
	}
	int numel(){
	    int result=1;
	    for(int i=0;i<shape.size();i++){
	        result*=shape[i];
	    }
	    return result;
	}
	vector<int> getshape(){
	    return shape;
	}
	vector<int> getstrides(){
	    return strides;
	}
	bool iscontiguous(){
	    bool returnbool=false;
	    if(strides.empty()) return false;
	    if(strides.back()==1){
	        returnbool=true;
	        for(size_t i=0;i+1<strides.size();i++){
	            if(strides[i] != strides[i+1]*shape[i+1]){
	                returnbool=false;
	                break;
	            }
	        }
	    }
	    return returnbool;
	}
};
}

#endif // TENSORIMPL_HPP