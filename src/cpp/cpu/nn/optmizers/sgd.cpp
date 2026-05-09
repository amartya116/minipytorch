#include <iostream>
#include <vector>
#include <immintrin.h>
#include "src\cpp\autograd\masternode.hpp"

using namespace std;
using namespace mylib::autograd;
void optmizerSGD(float learningrate,Tensornode* thingtoptimizenode){
    int i=0;
    for(;i<thingtoptimizenode->size;i=i+8){
        __m256 A=_mm256_loadu_ps(thingtoptimizenode->data + i);
        __m256 B=_mm256_loadu_ps(thingtoptimizenode->grad + i);
        __m256 C=_mm256_mul_ps(B,_mm256_set1_ps(learningrate));
        __m256 result = _mm256_sub_ps(A, C);
        _mm256_storeu_ps(thingtoptimizenode->data + i, result);
    }
    for(;i<thingtoptimizenode->size;i++){
        thingtoptimizenode->data[i]=thingtoptimizenode->data[i]-learningrate*thingtoptimizenode->grad[i];
    }
}