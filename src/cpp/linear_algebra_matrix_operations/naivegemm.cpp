#include <iostream>
#include <vector>

using namespace std;
void transpose(float* input,int K,int N,float* B_T) {
	for(int i=0; i<N; i++) {
		for(int j=0; j<K; j++) {
			B_T[j*N + i] = input[i*K + j];
		}
	}
}

void gemmcacheaware(float* A, float* B, float* C,
                    int M, int K, int N,
                    int mm, int kk, int nn) {

	std::fill(C, C + M * N, 0.0f);
	float* B_T = new float[K*N];
	transpose(B, K, N, B_T);
	for (int tilem = 0; tilem < M; tilem += mm) {
		for (int tilen = 0; tilen < N; tilen += nn) {
			float c00=0,c01=0,c10=0,c11=0;
			for (int tilek = 0; tilek < K; tilek += kk) {
				int M_end = std::min(tilem + mm, M);
				int K_end = std::min(tilek + kk, K);
				int N_end = std::min(tilen + nn, N);
				for (int i = tilem; i < M_end; i=i+2) {
					for (int j = tilen; j < N_end; j=j+2) {
						for (int k = tilek; k < K_end; k++) {
							float a0=A[i*K+k];
							float a1=A[(i+1)*K+k];
							float b0=B_T[j*K+k];
							float b1=B_T[(j+1)*K+k];
							c00 += a0 * b0;
							c01 += a0 * b1;
							c10 += a1 * b0;
							c11 += a1 * b1;

						}
						C[i*N+j]   = c00;
						C[i*N+j+1] = c01;
						C[(i+1)*N+j] = c10;
						C[(i+1)*N+j+1] = c11;
					}

				}
			}
		}
	}
	delete[] B_T;
}
int main() {
	int M = 4, N = 4, K = 4;
	int mm = 2, nn = 2, kk = 2;

	float A[16] = {
		1,  2,  3,  4,
		5,  6,  7,  8,
		1,  0,  1,  0,
		2,  1,  2,  1
	};

	float B[16] = {
		1,  0,  1,  0,
		0,  1,  0,  1,
		1,  2,  3,  4,
		2,  1,  2,  1
	};

	float C[16];

	gemmcacheaware(A, B, C, M, K, N, mm, kk, nn);

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			cout << C[i * N + j] << " ";
		}
		cout << endl;
	}

	return 0;
}