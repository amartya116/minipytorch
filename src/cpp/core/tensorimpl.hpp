#include <stdio.h>
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
	bool iscontigous(){
	    bool returnbool=false;
	    if(strides[strides.size()-1]==1){
	    for(int i=0;i<strides.size()-1;i++){
	        if(strides[i]==strides[i+1]*shape[i+1]){
	            returnbool=true;
	        }
	        else{
	            return false;
	        }
	    }}
	    else{
	        return false;
	    }
	    return returnbool;
	}
	vector<int> getshape(){
	    return shape;
	}
	vector<int> getstride(){
	    return strides;
	}
};
}