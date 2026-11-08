 #include <iostream>
#include <stdio.h>
#include <vector>
#include <functional>
#include <unordered_set>
#include <cstring>
#include "../../../core/tensorimpl.hpp"
#include "../../../core/dtype.hpp"
#include "../Initialization/initneu.hpp"
#include "../../../autograd/autogradnodes/addnode.cpp"
#include "../../../autograd/autogradnodes/mulnode.cpp"
#include "../../../autograd/autogradnodes/lossfunctionnode.cpp"
#include "../../../autograd/AutogradEngine.cpp"
#include <immintrin.h>
#include <string>
#include <random>
using namespace std;
using namespace mylib::core;
using namespace mylib::tensor;

TensorImpl* linear(int batch, int input_features, int output_features, Dtypes dtype, device deviceType, int level){
        //resize data here

    int inputnodesize = batch * input_features;
    int weightnodesize = input_features * output_features;
    int biasaddsize = output_features;

    //grads resize here
    int inputnodegradsize = batch * input_features;
    int weightnodegradsize = input_features * output_features;
    int biasaddgradsize = output_features;

    auto inputnodesizeStorage = make_shared<Storage>(inputnodesize * sizeof(float), dtype, deviceType);
    auto inputnodesizeGradStorage = make_shared<Storage>(inputnodegradsize * sizeof(float), dtype, deviceType);
    auto weightnodeStorage = make_shared<Storage>(weightnodesize * sizeof(float), dtype, deviceType);
    auto weightnodedGradStorage = make_shared<Storage>(weightnodegradsize * sizeof(float), dtype, deviceType);
    auto biasaddStorage = make_shared<Storage>(biasaddsize * sizeof(float), dtype, deviceType);
    auto biasaddGradStorage = make_shared<Storage>(biasaddgradsize * sizeof(float), dtype, deviceType);

    float* inputData = static_cast<float*>(inputnodesizeStorage->data);
    float* weightData = static_cast<float*>(weightnodeStorage->data);
    float* biasData = static_cast<float*>(biasaddStorage->data);

    for(int i=0; i<inputnodesize; ++i){
        inputData[i] = 1.0f + i;
    }
    for(int i=0; i<weightnodesize; ++i){
        weightData[i] = static_cast<float>(he_uniform(input_features));
    }
    for(int i=0; i<biasaddsize; ++i){
        biasData[i] = 0.0f;
    }
    // pooled blocks are reused without being cleared, so zero the leaf grads
    memset(inputnodesizeGradStorage->data, 0, inputnodegradsize * sizeof(float));
    memset(weightnodedGradStorage->data, 0, weightnodegradsize * sizeof(float));
    memset(biasaddGradStorage->data, 0, biasaddgradsize * sizeof(float));

    auto noBackward = [](TensorImpl*) {};
    TensorImpl *inputnode = new TensorImpl(inputnodesizeStorage, inputnodesizeGradStorage, {batch, input_features}, {input_features, 1}, 0, noBackward, "null", level, false, {},false);
    TensorImpl *weightnode = new TensorImpl(weightnodeStorage, weightnodedGradStorage, {input_features, output_features}, {output_features, 1}, 0, noBackward, "null", 0, true, {},true);
    TensorImpl *biasaddnode = new TensorImpl(biasaddStorage, biasaddGradStorage, {output_features}, {1}, 0, noBackward, "null", 0, true, {},true);
    TensorImpl *weightmulintermideatenode = matmul(inputnode, weightnode, dtype);
    TensorImpl *linearnode = addbias(weightmulintermideatenode, biasaddnode, dtype);
    return linearnode;
}

int main() {
    cout << "Starting main..." << endl;
    int batch = 2;
    int input_features = 4;
    int output_features = 3;
    float expected[6] = {
    1.9f, 3.7f, 5.5f,
    3.7f, 8.2f, 12.7f
};

    cout << "Calling linear()..." << endl;
    TensorImpl* result = linear(batch, input_features, output_features, Dtypes::float32, device::CPU, 0);
    cout << "Linear done, calling lossfunctionnode()..." << endl;
    TensorImpl* loss= lossfunctionnode(result,expected,Dtypes::float32);
    cout << "Loss done, calling backward()..." << endl;
    zerograd(loss);
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
    freetape();   // intermediates only, leaves stay
    return 0;
}
