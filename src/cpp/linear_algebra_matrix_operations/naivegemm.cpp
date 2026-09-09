#include <iostream>
#include <vector>
#include <immintrin.h>

using namespace std;
//architcture -> 
//tile-> pack -> microkernal simd
//C is m x n 
//A is m x k
//B is k x n
void microkernal_6x16(float *A,float *B,float *C,int n,int k){
	__m256 c00=_mm256_setzero_ps();
	__m256 c01=_mm256_setzero_ps();
	__m256 c02=_mm256_setzero_ps();
	__m256 c03=_mm256_setzero_ps();
	__m256 c04=_mm256_setzero_ps();
	__m256 c05=_mm256_setzero_ps();
	__m256 c06=_mm256_setzero_ps();
	__m256 c07=_mm256_setzero_ps();
	__m256 c08=_mm256_setzero_ps();
	__m256 c09=_mm256_setzero_ps();
	__m256 c10=_mm256_setzero_ps();
	__m256 c11=_mm256_setzero_ps();
	for(int iteratorK=0;iteratorK<k;iteratorK++){
		__m256 a00=_mm256_loadu_ps(&B[iteratorK*n]);
		__m256 a01=_mm256_loadu_ps(&B[iteratorK*n+8]);
		__m256 b00=_mm256_set1_ps(A[0*k+iteratorK]);
		__m256 b01=_mm256_set1_ps(A[1*k+iteratorK]);
		c00=_mm256_fmadd_ps(a00,b00,c00);
		c01=_mm256_fmadd_ps(a01,b00,c01);
		c02=_mm256_fmadd_ps(a00,b01,c02);
		c03=_mm256_fmadd_ps(a01,b01,c03);
		__m256 b02=_mm256_set1_ps(A[2*k+iteratorK]);
		__m256 b03=_mm256_set1_ps(A[3*k+iteratorK]);
		c04=_mm256_fmadd_ps(a00,b02,c04);
		c05=_mm256_fmadd_ps(a01,b02,c05);
		c06=_mm256_fmadd_ps(a00,b03,c06);
		c07=_mm256_fmadd_ps(a01,b03,c07);
		__m256 b04=_mm256_set1_ps(A[4*k+iteratorK]);
		__m256 b05=_mm256_set1_ps(A[5*k+iteratorK]);
		c08=_mm256_fmadd_ps(a00,b04,c08);
		c09=_mm256_fmadd_ps(a01,b04,c09);
		c10=_mm256_fmadd_ps(a00,b05,c10);
		c11=_mm256_fmadd_ps(a01,b05,c11);
		}
		_mm256_storeu_ps(&C[0*n + 0], c00);   
		_mm256_storeu_ps(&C[0*n + 8], c01);   // row 0
		_mm256_storeu_ps(&C[1*n + 0], c02);  
		_mm256_storeu_ps(&C[1*n + 8], c03);   // row 1
		_mm256_storeu_ps(&C[2*n + 0], c04);  
		_mm256_storeu_ps(&C[2*n + 8], c05);   // row 2
		_mm256_storeu_ps(&C[3*n + 0], c06);  
		_mm256_storeu_ps(&C[3*n + 8], c07);   // row 3
		_mm256_storeu_ps(&C[4*n + 0], c08);  
		_mm256_storeu_ps(&C[4*n + 8], c09);   // row 4
		_mm256_storeu_ps(&C[5*n + 0], c10);  
		_mm256_storeu_ps(&C[5*n + 8], c11);   // row 5

	}
	
	

void cacheblocking(){
	 
}