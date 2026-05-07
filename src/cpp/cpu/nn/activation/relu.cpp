#include <iostream>
#include <vector>
#include <immintrin.h>
using namespace std;

float* relu(float* A,int size){
    __m256 simd0=_mm256_setzero_ps();
    float* C;
    int i=0;
    for(;i<size-7;i=i+8){
        __m256 simdA=_mm256_loadu_ps(&A[i]);
        _mm256_storeu_ps(&C[i],_mm256_max_ps(simdA,simd0));
    }
    for(;i<size;i++){
        C[i]=A[i]>0?A[i]:0;
    }
    return C;
}
