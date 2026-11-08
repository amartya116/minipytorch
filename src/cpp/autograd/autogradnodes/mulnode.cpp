#include <iostream>
#include <vector>
#include <string>
#include <immintrin.h>
#include "../../core/tensorimpl.hpp"
#include "../../core/dtype.hpp"
#include "../../linear_algebra_matrix_operations/simdmatmul.hpp"
using namespace mylib::tensor;
using namespace mylib::core;
using namespace std;
void mulbackward(TensorImpl* NodeInput);

TensorImpl* matmul(TensorImpl *input1, TensorImpl *input2, Dtypes dtype) {
    int M = input1->shape[0];
    int K = input1->shape[1];
    int N = input2->shape[1];
    int resultNumel = M * N;
    int resultBytes = resultNumel * sizeof(float);

    auto resultStorage = make_shared<Storage>(resultBytes, input1->storage->dtype, input1->storage->device);
    auto resultGradStorage = make_shared<Storage>(resultBytes, input1->storage->dtype, input1->storage->device);

    float* input1Data = static_cast<float*>(input1->storage->data);
    float* input2Data = static_cast<float*>(input2->storage->data);
    float* resultData = static_cast<float*>(resultStorage->data);

    matmulsimdgemmcacheaware(input1Data, input2Data, resultData, M, K, N);

    TensorImpl* result = new TensorImpl(resultStorage, resultGradStorage, {M, N}, {N, 1}, 0, mulbackward, "mulbackward", 0, true, {input1, input2}, false);
    tape.push_back(result);
    return result;
}

void mulbackward(TensorImpl* NodeInput) {
    float* nodeGradData = static_cast<float*>(NodeInput->storageforgrad->data);
    float* parent0GradData = static_cast<float*>(NodeInput->parents[0]->storageforgrad->data);
    float* parent1GradData = static_cast<float*>(NodeInput->parents[1]->storageforgrad->data);
    float* parent0Data = static_cast<float*>(NodeInput->parents[0]->storage->data);
    float* parent1Data = static_cast<float*>(NodeInput->parents[1]->storage->data);
    int M = NodeInput->parents[0]->shape[0];
    int K = NodeInput->parents[0]->shape[1];
    int N = NodeInput->parents[1]->shape[1];

    // C = A*B  (A MxK, B KxN, dC MxN)
    // dA = dC * B^T  (MxK)
    // dB = A^T * dC  (KxN)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                parent0GradData[i * K + k] += nodeGradData[i * N + j] * parent1Data[k * N + j];
                parent1GradData[k * N + j] += parent0Data[i * K + k] * nodeGradData[i * N + j];
            }
        }
    }
}
