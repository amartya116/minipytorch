 #include <iostream>
#include <stdio.h>
#include <vector>
#include <functional>
#include <unordered_set>
#include "src\cpp\core\tensorimpl.hpp"
#include "src\cpp\core\dtype.hpp"
#include "src\cpp\autograd\autogradnodes\addnode.cpp"
#include "src\cpp\autograd\autogradnodes\mulnode.cpp"
#include <immintrin.h>
#include <string> 
#include <random>
using namespace std;
using namespace mylib::tensor;

TensorImpl* linear(int inputbatch,int output,enum Dtypes dtype,string device,int level){
        //resize data her
    
    int inputnodesize=inputbatch*output;
    int weightnodesize=output*output; 
    int biasaddsize=inputbatch*output;
    int weightmulintermideatesize=inputbatch*output;
    int linearsize=inputbatch*output;
    
    //grads resize here 
    int inputnodegradsize=inputbatch*output;
    int weightnodegradsize=output*output; 
    int biasaddgradsize=inputbatch*output;
    int weightmulintermideategradsize=inputbatch*output;
    int lineargradsize=inputbatch*output; 


    auto inputnodesizeStorage = make_shared<Storage>(inputnodesize*sizeof(float),dtype, device);
    auto inputnodesizeGradStorage = make_shared<Storage>(inputnodegradsize*sizeof(float),dtype, device);
    auto weightnodeStorage = make_shared<Storage>(weightnodesize*sizeof(float),dtype, device);
    auto weightnodedGradStorage = make_shared<Storage>(weightnodegradsize*sizeof(float),dtype, device);
    auto biasaddStorage = make_shared<Storage>(biasaddsize*sizeof(float),dtype, device);
    auto biasaddGradStorage = make_shared<Storage>(biasaddgradsize*sizeof(float),dtype, device);
     auto weightmulintermideateStorage = make_shared<Storage>(weightmulintermideatesize*sizeof(float),dtype, device);
    auto weightmulintermideateGradStorage = make_shared<Storage>(weightmulintermideategradsize*sizeof(float),dtype, device);
     auto linearStorage = make_shared<Storage>(linearsize*sizeof(float), dtype, device);
    auto linearGradStorage = make_shared<Storage>(lineargradsize*sizeof(float), dtype, device);

    auto noBackward = [](TensorImpl*) {};
    TensorImpl *inputnode = new TensorImpl(inputnodesizeStorage, inputnodesizeGradStorage, {inputbatch,output}, {inputbatch,output}, 0, noBackward, "null", level, false, {});
    TensorImpl *weightnode = new TensorImpl(weightnodeStorage, weightnodedGradStorage, {output,output}, {output,output}, 0, noBackward, "null", 0, false, {});
    TensorImpl *biasaddnode = new TensorImpl(biasaddStorage, biasaddGradStorage, {inputbatch,output}, {inputbatch,output}, 0, noBackward, "null", 0, false, {});
    TensorImpl *weightmulintermideatenode = matmul(inputnode,weightnode,dtype);
    TensorImpl *linearnode = add(weightmulintermideatenode,biasaddnode,dtype);
    return linearnode;
}
