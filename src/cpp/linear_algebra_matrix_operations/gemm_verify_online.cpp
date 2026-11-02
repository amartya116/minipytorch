#include <cstdio>
#include <vector>

// One 6x16 tile of C.
// Ap[k*6 + r]  = A[row r][k]    (6 floats per k)
// Bp[k*16 + c] = B[k][col c]    (16 floats per k)
// C points at the tile's top-left, row stride ldc.
// first: true -> C = A*B, false -> C += A*B
void kernel_6x16(int kc, const float* Ap, const float* Bp, float* C, int ldc, bool first) {
    float acc[6][16];
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 16; ++c)
            acc[r][c] = first ? 0.0f : C[r * ldc + c];
    for (int k = 0; k < kc; ++k)
        for (int r = 0; r < 6; ++r) 
            for (int c = 0; c < 16; ++c)
                acc[r][c] += Ap[k * 6 + r] * Bp[k * 16 + c];
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 16; ++c)
            C[r * ldc + c] = acc[r][c];
}

// ======================= YOUR CODE =======================
void pack_A(const float* A, int K, int cr, int cc, int mc, int kc, float* Ap) {
    for(int ri=cr;ri<cr+mc;ri+=6){
        for(int ci=cc;ci<cc+kc;ci++){
            int p = (ri - cr);  
            int k = ci - cc   ; 
            Ap[p*kc+k*6]=A[K*ri+ci];
            Ap[p*kc+k*6+1]=A[K*(ri+1)+ci];
            Ap[p*kc+k*6+2]=A[K*(ri+2)+ci];
            Ap[p*kc+k*6+3]=A[K*(ri+3)+ci];
            Ap[p*kc+k*6+4]=A[K*(ri+4)+ci];
            Ap[p*kc+k*6+5]=A[K*(ri+5)+ci];
        }
    }
}

void pack_B(const float* B, int N, int cr, int cc, int kc, int nc, float* Bp) {
    for(int ri=cr;ri<cr+kc;ri++){
        for(int ci=cc;ci<cc+nc;ci+=16){ 
            for(int cci=0;cci<16;cci++){
            int p = (ri - cr);  
            int k = ci - cc   ; 
            Bp[k*kc+p*16+cci]=B[N*ri+ci+cci];
        }
    }}
}

void gemm_blocked(int M, int N, int K, const float* A, const float* B, float* C,
                  int MC, int KC, int NC) {
}
// =========================================================

void print(const char* name, const float* X, int rows, int cols) {
    printf("\n%s:\n", name);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) printf("%7.0f", X[r * cols + c]);
        printf("\n");
    }
}

int main() {
    int M = 12, K = 8, N = 32;
    int MC = 12, KC = 4, NC = 32;

    std::vector<float> A(M * K), B(K * N), C(M * N, 0), Cexp(M * N, 0);
    for (int r = 0; r < M; ++r) for (int k = 0; k < K; ++k) A[r * K + k] = 10 * r + k;
    for (int k = 0; k < K; ++k) for (int j = 0; j < N; ++j) B[k * N + j] = 100 * k + j;

    std::vector<float> Ap(MC * KC, -1), Bp(KC * NC, -1);
    pack_A(A.data(), K, 0, 0, MC, KC, Ap.data());
    pack_B(B.data(), N, 0, 0, KC, NC, Bp.data());
    print("packed A, first block (one line = one k)", Ap.data(), MC * KC / 6, 6);
    print("packed B, first block (one line = one k)", Bp.data(), KC * NC / 16, 16);

    gemm_blocked(M, N, K, A.data(), B.data(), C.data(), MC, KC, NC);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k)
                Cexp[i * N + j] += A[i * K + k] * B[k * N + j];

    print("C yours", C.data(), M, N);
    print("C expected", Cexp.data(), M, N);
}
