#include <iostream>
#include <vector>
#include <immintrin.h>
#include "../../../core/tensorimpl.hpp"
using namespace std;
using namespace mylib::tensor;
void optmizerSGD(float learningrate,TensorImpl* thingtoptimizenode){
    int i=0;
    float* nodeData = static_cast<float*>(thingtoptimizenode->storageforgrad->data);
    float* nodeGradData = static_cast<float*>(thingtoptimizenode->storageforgrad->data);
    for(;i<thingtoptimizenode->storage->nbytes-7;i=i+8){
        __m256 A=_mm256_loadu_ps(nodeData + i);
        __m256 B=_mm256_loadu_ps(nodeGradData + i);
        __m256 C=_mm256_mul_ps(B,_mm256_set1_ps(learningrate));
        __m256 result = _mm256_sub_ps(A, C);
        _mm256_storeu_ps(nodeData + i, result);
    }
    for(;i<thingtoptimizenode->storage->nbytes;i++){
        nodeData[i]=nodeData[i]-learningrate*nodeGradData[i];
    }
}