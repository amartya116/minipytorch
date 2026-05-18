#include <iostream>
#include <vector>
#include <string>
#include <immintrin.h>
#include "src\cpp\core\tensorimpl.hpp"
#include "src\cpp\core\dtype.hpp"
#include "src\cpp\linear_algebra_matrix_operations\simdmatmul.hpp"
using namespace mylib::tensor;
using namespace mylib::core;
using namespace std;

TensorImpl* matmul(TensorImpl *input1, TensorImpl *input2,enum Dtypes dtype) {
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

    TensorImpl* result = new TensorImpl(resultStorage, resultGradStorage, {M, N}, {N, 1}, 0, mulbackward, "mulbackward", 0, true, {input1, input2});
    return result;
}

void mulbackward(TensorImpl* NodeInput) {
    int i = 0;
    float* nodeGradData = static_cast<float*>(NodeInput->storageforgrad->data);
    float* parent0GradData = static_cast<float*>(NodeInput->parents[0]->storageforgrad->data);
    float* parent1GradData = static_cast<float*>(NodeInput->parents[1]->storageforgrad->data);
    float* parent0Data = static_cast<float*>(NodeInput->parents[0]->storage->data);
    float* parent1Data = static_cast<float*>(NodeInput->parents[1]->storage->data);
    int numel = NodeInput->parents[0]->shape[0] * NodeInput->parents[0]->shape[1];

    for (; i < numel - 7; i += 8) {
        _mm256_storeu_ps(parent0GradData + i,
            _mm256_add_ps(_mm256_loadu_ps(parent0GradData + i),
                          _mm256_mul_ps(_mm256_loadu_ps(parent1Data + i), _mm256_loadu_ps(nodeGradData + i))));
        _mm256_storeu_ps(parent1GradData + i,
            _mm256_add_ps(_mm256_loadu_ps(parent1GradData + i),
                          _mm256_mul_ps(_mm256_loadu_ps(parent0Data + i), _mm256_loadu_ps(nodeGradData + i))));
    }

    for (; i < numel; i++) {
        parent0GradData[i] += parent1Data[i] * nodeGradData[i];
        parent1GradData[i] += parent0Data[i] * nodeGradData[i];
    }
}
