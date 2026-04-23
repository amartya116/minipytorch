#include <iostream>
using namespace std;

void gemmcacheaware(float* A, float* B, float* C,
                    int M, int K, int N,
                    int mm, int kk, int nn) {

	std::fill(C, C + M * N, 0.0f);

	for (int tilem = 0; tilem < M; tilem += mm) {
		for (int tilek = 0; tilek < K; tilek += kk) {
			for (int tilen = 0; tilen < N; tilen += nn) {

				int M_end = std::min(tilem + mm, M);
				int K_end = std::min(tilek + kk, K);
				int N_end = std::min(tilen + nn, N);

				for (int i = tilem; i < M_end; i++) {
					for (int k = tilek; k < K_end; k++) {

						float a = A[i * K + k];

						for (int j = tilen; j < N_end; j++) {
							C[i * N + j] += a * B[k * N + j];
						}
					}
				}
			}
		}
	}
}
int main() {
	int M,N,K,mm,nn,kk;
	float A[9]= {1 ,2, 3,
 4, 5, 6,
 7, 8 ,9};
	float B[9]= {9, 8, 7,
 6, 5 ,4,
 3, 2, 1};
	float C[9];
	gemmcacheaware(A,B,C,M = 3,K = 3,N = 3,mm = 2, kk = 2, nn = 2);
	return 0;
}
