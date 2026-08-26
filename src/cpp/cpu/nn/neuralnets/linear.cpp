 #include <iostream>
#include <stdio.h>
#include <vector>
#include <functional>
#include <unordered_set>
#include "../../../core/tensorimpl.hpp"
#include "../../../core/dtype.hpp"
#include "../../../autograd/autogradnodes/addnode.cpp"
#include "../../../autograd/autogradnodes/mulnode.cpp"
#include "../../../autograd/autogradnodes/lossfunctionnode.cpp"
#include "../../../autograd/AutogradEngine.hpp"
#include <immintrin.h>
#include <string>
#include <random>
using namespace std;
using namespace mylib::core;
using namespace mylib::tensor;

TensorImpl* linear(int inputbatch, int output, Dtypes dtype, device deviceType, int level){
        //resize data here
    
    int inputnodesize = inputbatch * output;
    int weightnodesize = output * output; 
    int biasaddsize = inputbatch * output;
    int weightmulintermideatesize = inputbatch * output;
    int linearsize = inputbatch * output;
    
    //grads resize here 
    int inputnodegradsize = inputbatch * output;
    int weightnodegradsize = output * output; 
    int biasaddgradsize = inputbatch * output;
    int weightmulintermideategradsize = inputbatch * output;
    int lineargradsize = inputbatch * output; 

    auto inputnodesizeStorage = make_shared<Storage>(inputnodesize * sizeof(float), dtype, deviceType);
    auto inputnodesizeGradStorage = make_shared<Storage>(inputnodegradsize * sizeof(float), dtype, deviceType);
    auto weightnodeStorage = make_shared<Storage>(weightnodesize * sizeof(float), dtype, deviceType);
    auto weightnodedGradStorage = make_shared<Storage>(weightnodegradsize * sizeof(float), dtype, deviceType);
    auto biasaddStorage = make_shared<Storage>(biasaddsize * sizeof(float), dtype, deviceType);
    auto biasaddGradStorage = make_shared<Storage>(biasaddgradsize * sizeof(float), dtype, deviceType);
    auto weightmulintermideateStorage = make_shared<Storage>(weightmulintermideatesize * sizeof(float), dtype, deviceType);
    auto weightmulintermideateGradStorage = make_shared<Storage>(weightmulintermideategradsize * sizeof(float), dtype, deviceType);
    auto linearStorage = make_shared<Storage>(linearsize * sizeof(float), dtype, deviceType);
    auto linearGradStorage = make_shared<Storage>(lineargradsize * sizeof(float), dtype, deviceType);

    float* inputData = static_cast<float*>(inputnodesizeStorage->data);
    float* weightData = static_cast<float*>(weightnodeStorage->data);
    float* biasData = static_cast<float*>(biasaddStorage->data);

    for(int i=0; i<inputnodesize; ++i){
        inputData[i] = 1.0f + i;
    }
    for(int i=0; i<weightnodesize; ++i){
        weightData[i] = 0.1f * (i + 1);
    }
    for(int i=0; i<biasaddsize; ++i){
        biasData[i] = 0.5f;
    }

    auto noBackward = [](TensorImpl*) {};
    TensorImpl *inputnode = new TensorImpl(inputnodesizeStorage, inputnodesizeGradStorage, {inputbatch, output}, {output, 1}, 0, noBackward, "null", level, false, {},false);
    TensorImpl *weightnode = new TensorImpl(weightnodeStorage, weightnodedGradStorage, {output, output}, {output, 1}, 0, noBackward, "null", 0, true, {},true);
    TensorImpl *biasaddnode = new TensorImpl(biasaddStorage, biasaddGradStorage, {inputbatch, output}, {output, 1}, 0, noBackward, "null", 0, true, {},true);
    TensorImpl *weightmulintermideatenode = matmul(inputnode, weightnode, dtype);
    TensorImpl *linearnode = add(weightmulintermideatenode, biasaddnode, dtype);
    return linearnode;
}

int main() {
    cout << "Starting main..." << endl;
    int batch = 2;
    int output = 3;
    float expected[6] = {
    1.9f, 3.7f, 5.5f,
    3.7f, 8.2f, 12.7f
};

    cout << "Calling linear()..." << endl;
    TensorImpl* result = linear(batch, output, Dtypes::float32, device::CPU, 0);
    cout << "Linear done, calling lossfunctionnode()..." << endl;
    TensorImpl* loss= lossfunctionnode(result,expected,Dtypes::float32);
    cout << "Loss done, calling backward()..." << endl;
    backward(loss);
    cout << "Backward done (skipped), calling optmizerengine()..." << endl;
    optmizerengine(loss);
    cout << "Optimizer done, getting results..." << endl;
    int rows = result->shape[0];
    int cols = result->shape[1]; 
    float* resultData = static_cast<float*>(result->storage->data);

    cout << "Result shape: [" << rows << ", " << cols << "]\n";
    cout << "Result values:\n";
    for(int i = 0; i < rows * cols; ++i) {
        cout << resultData[i];
        if ((i + 1) % cols == 0) {
            cout << "\n";
        } else {
            cout << " ";
        }
    }
    return 0;
}
