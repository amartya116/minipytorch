#ifndef TENSORIMPL_HPP
#define TENSORIMPL_HPP
#include <string>
#include <cstddef>
#include <memory>
#include <vector>
#include <functional>
#include "storage.hpp"

namespace mylib::tensor
{
struct TensorImpl
{
	shared_ptr<Storage> storage;
	shared_ptr<Storage> storageforgrad;
	int level;
	bool requiregrad;
    vector<TensorImpl*> parents;
	float singlevaluedata;
	vector<int> shape;
	vector<int> strides;
	int offset;
	string backwardfnname;
	mylib::core::Dtypes dtype;
	bool tooptmize;
	function<void(TensorImpl*)> backwardfunction;
	mylib::core::device device;
	TensorImpl(shared_ptr<Storage> storage,
		shared_ptr<Storage> storageforgrad, vector<int> shape, vector<int> strides, int offset, 
		function<void(TensorImpl*)> backwardfunction,string backwardfnname,int level,bool requiregrad,vector<TensorImpl*> parents,bool tooptmize){
	    this->storage=storage;
		this->storageforgrad=storageforgrad;
	    this->shape=shape;
		this->level=level;
		this->requiregrad=requiregrad;
		this->backwardfunction=backwardfunction;
		this->backwardfnname=backwardfnname;
	    this->strides=strides;
		this->tooptmize=tooptmize;
	    this->offset=offset;
	    this->dtype=storage->dtype;
	    this->device=storage->device;
	    this->parents=parents;
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