#include <iostream>
#include <vector>
#include <immintrin.h>
using namespace std;

void transpose(const float* B, int K, int N,float* B_T) {
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            B_T[j * K + k] = B[k * N + j];
        }
    }
}
void matmulsimdgemmcacheaware(float* A, float* B, float* C,
                    int M, int K, int N) {

    
	std::fill(C, C + M * N, 0.0f);
	float* B_T = new float[K*N];
	transpose(B, K, N, B_T);
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
            __m256 c0=_mm256_setzero_ps();
            int k=0;
				for(; k + 7 < K; k=k+8) {
                    __m256 a0=_mm256_loadu_ps(&A[i*K+k]);
                    __m256 b0=_mm256_loadu_ps(&B_T[j*K+k]);
                    c0=_mm256_fmadd_ps(a0,b0,c0);
                }
            float temp[8];
            _mm256_storeu_ps(temp, c0);
            float result = temp[0] + temp[1] + temp[2] + temp[3]
                         + temp[4] + temp[5] + temp[6] + temp[7];

            // remainder loop
            for (; k < K; k++) {
                result += A[i * K + k] * B_T[j * K + k];
            }

            C[i * N + j] = result;
			}
		}
}
int main() {
    int M = 8, N = 8, K = 8;


    float A[64];
    float B[64];
    float C[64];

    // Fill A with predictable values (1..64)
    int val = 1;
    for (int i = 0; i < 64; i++) {
        A[i] = (float)val++;
    }

    // Identity matrix for B
    for (int i = 0; i < 64; i++) {
        B[i] = 0.0f;
    }
    for (int i = 0; i < 8; i++) {
        B[i * 8 + i] = 1.0f;
    }

    // Call your function
    matmulsimdgemmcacheaware(A, B, C, M, K, N);

    // Print result
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            cout << C[i * N + j] << " ";
        }
        cout << endl;
    }

    return 0;
}