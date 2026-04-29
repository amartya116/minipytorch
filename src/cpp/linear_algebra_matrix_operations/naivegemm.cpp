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
			float c00=0,c01=0,c02=0,c03=0,c10=0,c11=0,c12=0,c13=0,c20=0,c21=0,c22=0,c23=0,c30=0,c31=0,c32=0,c33=0;
			for (int tilek = 0; tilek < K; tilek += kk) {
				int M_end = std::min(tilem + mm, M);
				int K_end = std::min(tilek + kk, K);
				int N_end = std::min(tilen + nn, N);
				for (int i = tilem; i < M_end; i=i+4) {
					for (int j = tilen; j < N_end; j=j+4) {
						for (int k = tilek; k < K_end; k++) {
							float a0=A[i*K+k];
							float a1=A[(i+1)*K+k];
							float a2=A[(i+2)*K+k];
							float a3=A[(i+3)*K+k];
							float b0=B_T[j*K+k];
							float b1=B_T[(j+1)*K+k];
							float b2=B_T[(j+2)*K+k];
							float b3=B_T[(j+3)*K+k];
							c00 += a0 * b0;
							c01 += a0 * b1;
							c02 += a0 * b2;
							c03 += a0 * b3;
							c10 += a1 * b0;
							c11 += a1 * b1;
							c12 += a1 * b2;
							c13 += a1 * b3;
							c20 += a2 * b0;
							c21 += a2 * b1;
							c22 += a2 * b2;
							c23 += a2 * b3;
							c30 += a3 * b0;
							c31 += a3 * b1;
							c32 += a3 * b2;
							c33 += a3 * b3;

						}
						C[i*N+j]   = c00;
						C[i*N+j+1] = c01;
						C[i*N+j+2] = c02;
						C[i*N+j+3] = c03;
						C[(i+1)*N+j] = c10;
						C[(i+1)*N+j+1] = c11;
						C[(i+1)*N+j+2] = c12;
						C[(i+1)*N+j+3] = c13;
						C[(i+2)*N+j] = c20;
						C[(i+2)*N+j+1] = c21;
						C[(i+2)*N+j+2] = c22;
						C[(i+2)*N+j+3] = c23;
						C[(i+3)*N+j] = c30;
						C[(i+3)*N+j+1] = c31;
						C[(i+3)*N+j+2] = c32;
						C[(i+3)*N+j+3] = c33;
					}

				}
			}
		}
	}
	delete[] B_T;
}
int main() {
	int M = 4, N = 4, K = 4;
	int mm = 4, nn = 4, kk = 2;

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