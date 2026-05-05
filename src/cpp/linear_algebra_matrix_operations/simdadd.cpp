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

int main(){
    int N = 1, K = 5;  // N*K = 5, less than 8
    float* A = new float[N*K];
    float* B = new float[N*K];
    float* C = new float[N*K];

    for(int i = 0; i < N*K; i++){
        A[i] = (float)i;
        B[i] = 1.0f;
    }

    simdadd(A, B, C, N,K);

    for(int i = 0; i < N*K; i++){
        cout << C[i] << " ";
    }
    cout << endl;

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}