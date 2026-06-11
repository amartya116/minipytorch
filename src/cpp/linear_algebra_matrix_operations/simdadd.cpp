 #include <iostream>
#include <immintrin.h>
using namespace std;

void simdadd(float* A,float* B,float* C,int N,int K){
    int total = N*K;
    int i = 0;
    for(; i + 7 < total; i += 8){
    __m256 a = _mm256_loadu_ps(&A[i]);
    __m256 b = _mm256_loadu_ps(&B[i]);
    _mm256_storeu_ps(&C[i], _mm256_add_ps(a, b));
}
    for(; i < total; i++){
    C[i] = A[i] + B[i];
}
}
void simdsub(float* inputA,float *inputB,float *C,int N,int M){
    int total = N*M;
    int i=0;
    for(;i<total-7;i=i+8){
    __m256 a=_mm256_loadu_ps(&inputA[i]);
    __m256 b=_mm256_loadu_ps(&inputB[i]);
    _mm256_storeu_ps(&C[i],_mm256_sub_ps(a,b));
}
    for(; i < total; i++){
    C[i] = inputA[i] - inputB[i];
}
}
void simdmul(float* inputA,float *inputB,float *C,int N,int M){
    int total = N*M;
    int i=0;
    for(;i<total-7;i=i+8){
    __m256 a=_mm256_loadu_ps(&inputA[i]);
    __m256 b=_mm256_loadu_ps(&inputB[i]);
    _mm256_storeu_ps(&C[i],_mm256_mul_ps(a,b));
}
    for(; i < total; i++){
    C[i] = inputA[i] * inputB[i];
}
}
void simddiv(float* inputA,float *inputB,float *C,int N,int M){
    int total = N*M;
    int i=0;
    for(;i<total-7;i=i+8){
    __m256 a=_mm256_loadu_ps(&inputA[i]);
    __m256 b=_mm256_loadu_ps(&inputB[i]);
    _mm256_storeu_ps(&C[i],_mm256_div_ps(a,b));
}
    for(; i < total; i++){
    C[i] = inputA[i] / inputB[i];
}
}