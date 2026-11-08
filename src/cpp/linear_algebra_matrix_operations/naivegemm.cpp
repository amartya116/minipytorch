#include <iostream>
#include <vector>
#include <immintrin.h>
#include <algorithm>
using namespace std;
//architcture -> 
//tile-> pack -> microkernal simd
//C is m x n 
//A is m x k
//B is k x n
// line 14: new signature. ldc = row stride of C, k = kcur, mr/nr = valid tile size, first = zero vs accumulate
void microkernal_6x16(const float *A,const float *B,float *C,int ldc,int k,int mr,int nr,bool first){
	// edge tile: work in a 6x16 scratch buffer so full-width stores can't run past C
	float tmp[6*16];
	float *Ct=C;
	int ld=ldc;
	if(mr<6||nr<16){
		Ct=tmp; ld=16;
		if(!first){
			for(int r=0;r<6;r++)
				for(int c=0;c<16;c++)
					tmp[r*16+c]=(r<mr&&c<nr)?C[r*ldc+c]:0.0f;
		}
	}
	// lines 15-26: zero on the first K block, otherwise load the existing C (C += A*B)
	__m256 c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11;
	if(first){
		c00=c01=c02=c03=c04=c05=c06=c07=c08=c09=c10=c11=_mm256_setzero_ps();
	}else{
		c00=_mm256_loadu_ps(&Ct[0*ld+0]); c01=_mm256_loadu_ps(&Ct[0*ld+8]);
		c02=_mm256_loadu_ps(&Ct[1*ld+0]); c03=_mm256_loadu_ps(&Ct[1*ld+8]);
		c04=_mm256_loadu_ps(&Ct[2*ld+0]); c05=_mm256_loadu_ps(&Ct[2*ld+8]);
		c06=_mm256_loadu_ps(&Ct[3*ld+0]); c07=_mm256_loadu_ps(&Ct[3*ld+8]);
		c08=_mm256_loadu_ps(&Ct[4*ld+0]); c09=_mm256_loadu_ps(&Ct[4*ld+8]);
		c10=_mm256_loadu_ps(&Ct[5*ld+0]); c11=_mm256_loadu_ps(&Ct[5*ld+8]);
	}
	for(int iteratorK=0;iteratorK<k;iteratorK++){
		// lines 28-29: B row (renamed b0x, since these are B)
		__m256 b00=_mm256_loadu_ps(&B[iteratorK*16]);
		__m256 b01=_mm256_loadu_ps(&B[iteratorK*16+8]);
		// lines 30-31, 36-37, 42-43: A[iteratorK*6 + r], matching the packed layout
		__m256 a0=_mm256_set1_ps(A[iteratorK*6+0]);
		__m256 a1=_mm256_set1_ps(A[iteratorK*6+1]);
		c00=_mm256_fmadd_ps(a0,b00,c00);
		c01=_mm256_fmadd_ps(a0,b01,c01);
		c02=_mm256_fmadd_ps(a1,b00,c02);
		c03=_mm256_fmadd_ps(a1,b01,c03);
		__m256 a2=_mm256_set1_ps(A[iteratorK*6+2]);
		__m256 a3=_mm256_set1_ps(A[iteratorK*6+3]);
		c04=_mm256_fmadd_ps(a2,b00,c04);
		c05=_mm256_fmadd_ps(a2,b01,c05);
		c06=_mm256_fmadd_ps(a3,b00,c06);
		c07=_mm256_fmadd_ps(a3,b01,c07);
		__m256 a4=_mm256_set1_ps(A[iteratorK*6+4]);
		__m256 a5=_mm256_set1_ps(A[iteratorK*6+5]);
		c08=_mm256_fmadd_ps(a4,b00,c08);
		c09=_mm256_fmadd_ps(a4,b01,c09);
		c10=_mm256_fmadd_ps(a5,b00,c10);
		c11=_mm256_fmadd_ps(a5,b01,c11);
	}
	_mm256_storeu_ps(&Ct[0*ld+0],c00); _mm256_storeu_ps(&Ct[0*ld+8],c01);
	_mm256_storeu_ps(&Ct[1*ld+0],c02); _mm256_storeu_ps(&Ct[1*ld+8],c03);
	_mm256_storeu_ps(&Ct[2*ld+0],c04); _mm256_storeu_ps(&Ct[2*ld+8],c05);
	_mm256_storeu_ps(&Ct[3*ld+0],c06); _mm256_storeu_ps(&Ct[3*ld+8],c07);
	_mm256_storeu_ps(&Ct[4*ld+0],c08); _mm256_storeu_ps(&Ct[4*ld+8],c09);
	_mm256_storeu_ps(&Ct[5*ld+0],c10); _mm256_storeu_ps(&Ct[5*ld+8],c11);
	if(Ct==tmp){
		for(int r=0;r<mr;r++)
			for(int c=0;c<nr;c++)
				C[r*ldc+c]=tmp[r*16+c];
	}
}
void cachepackingforrow(const float* A, int K, int cr, int cc, int mc, int kc, float* Ap){
	for(int ri=0;ri<mc;ri+=6){             // local offsets now; mc = mcur
		for(int k=0;k<kc;k++){             // kc = kcur
			for(int r=0;r<6;r++){
				// zero-pad rows past the block (i.e. past M)
				Ap[ri*kc+k*6+r]=(ri+r<mc)?A[K*(cr+ri+r)+cc+k]:0.0f;
			}
		}
	}
}
void cachepackingforcolumns(const float* B, int N, int cr, int cc, int kc, int nc, float* Bp){
	for(int p=0;p<kc;p++){                 // kc = kcur
		for(int ci=0;ci<nc;ci+=16){        // nc = ncur
			for(int cci=0;cci<16;cci++){
				Bp[ci*kc+p*16+cci]=(ci+cci<nc)?B[N*(cr+p)+cc+ci+cci]:0.0f;
			}
		}
	}
}
void gemmmatmul(int N,int M,int K,float *inputA,float* inputB,float* C){
	const int mc=96,kc=256,nc=1024;
	// line 90: sized for whole padded panels
	std::vector<float> packedinputA(((mc+5)/6*6)*kc), packedinputB(kc*((nc+15)/16*16));

	for(int column=0;column<N;column+=nc){
		int ncur=min(nc,N-column);
		for(int common=0;common<K;common+=kc){
			int kcur=min(kc,K-common);
			cachepackingforcolumns(inputB,N,common,column,kcur,ncur,packedinputB.data());
			for(int rows=0;rows<M;rows+=mc){
				int mcur=min(mc,M-rows);
				cachepackingforrow(inputA,K,rows,common,mcur,kcur,packedinputA.data());
				for(int packedcolumn=0;packedcolumn<ncur;packedcolumn+=16){
					for(int packedrow=0;packedrow<mcur;packedrow+=6){
						microkernal_6x16(&packedinputA[packedrow*kcur],&packedinputB[packedcolumn*kcur],
							&C[(rows+packedrow)*N+column+packedcolumn],N,kcur,
							min(6,mcur-packedrow),min(16,ncur-packedcolumn),common==0);
					}
				}
			}
		}
	}
}

// standalone test: compile with -DNAIVEGEMM_MAIN (off by default so gemm_benchmark.cpp can include this file)
#ifdef NAIVEGEMM_MAIN
#include <chrono>
#include <cmath>
#include <cstdio>

// plain triple loop, the reference answer
void referencegemm(int N,int M,int K,const float *A,const float *B,float *C){
	for(int i=0;i<M;i++)
		for(int j=0;j<N;j++){
			float sum=0.0f;
			for(int k=0;k<K;k++) sum+=A[i*K+k]*B[k*N+j];
			C[i*N+j]=sum;
		}
}


int main() {
    int n = 1024;

    vector<float> A(n * n, 1.0f);
    vector<float> B(n * n, 0.5f);
    vector<float> C(n * n), R(n * n);

    auto t0 = chrono::high_resolution_clock::now();
    gemmmatmul(n, n, n, A.data(), B.data(), C.data());
    auto t1 = chrono::high_resolution_clock::now();

    auto t2 = chrono::high_resolution_clock::now();
    referencegemm(n, n, n, A.data(), B.data(), R.data());
    auto t3 = chrono::high_resolution_clock::now();

    double simd_time = chrono::duration<double>(t1 - t0).count();
    double ref_time  = chrono::duration<double>(t3 - t2).count();
    double flops = 2.0 * n * n * n;

    printf("SIMD GEMM:      %.2f ms | %.1f GFLOP/s\n",
           simd_time * 1e3, flops / simd_time / 1e9);

    printf("Reference GEMM: %.2f ms | %.1f GFLOP/s\n",
           ref_time * 1e3, flops / ref_time / 1e9);

    printf("Speedup: %.2fx\n", ref_time / simd_time);

    return 0;
}
#endif
