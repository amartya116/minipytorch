 #include <iostream>
#include <immintrin.h>
using namespace std;

void simdadd(float* A,float* B,float* C,int N,int K){
    int simdres=N*K-N*K/8;
    for(int i=0;i<N*K;i=i+8){
    __m256 simda=_mm256_loadu_ps(&A[i]);
    __m256 simdb=_mm256_loadu_ps(&B[i]);
    __m256 simdc=_mm256_add_ps(simda,simdb);
    _mm256_storeu_ps(&C[i],simdc);

}
if(simdres!=0){
 for(int i=0;i<simdres;i++){
    C[i]=A[i]+B[i];
 }}
}

int main(){
    int N = 3, K = 6;  // N*K = 18
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