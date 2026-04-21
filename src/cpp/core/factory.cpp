#include <iostream>
#include "tensorimpl.hpp"
#include <vector>
#include <Tensor>
#include <atomic>
#include <memory>
namespace mylib::tensor{
    shared_ptr<TensorImpl> empty(vector<int> shape,mylib::core::Dtypes dtype, mylib::core::device dev){
        int numel;
            for(int i=0;i<shape.size();i++){ 
                numel*=shape[i];
            }
        int nbytes=shape.size()*mylib::core::itemsize(dtype);
        shared_ptr<Storage> storage=make_shared<Storage>(nbytes, dtype, dev);
        vector<int> strides;
        for(int i=0;i<shape.size();i++){
            strides[i]=shape[i+1]*shape[i+1];
        }
        shared_ptr<TensorImpl> result=make_shared<TensorImpl>(storage, shape, strides, 0);
        return result;
    }
    shared_ptr<TensorImpl> zero(vector<int> shape,mylib::core::Dtypes dtype, mylib::core::device dev){
       shared_ptr<TensorImpl> result= empty(shape,dtype,dev);
       int nbytes=shape.size()*mylib::core::itemsize(dtype);
       memset(result->storage->dataptr(), 0, nbytes);
       return result;
    }
    float* ones(vector<int> shape,mylib::core::Dtypes dtype, mylib::core::device dev){
       shared_ptr<TensorImpl> result= empty(shape,dtype,dev);
       int nbytes=shape.size()*mylib::core::itemsize(dtype);
        return static_cast<float*>(result->storage->dataptr());
    }
}  