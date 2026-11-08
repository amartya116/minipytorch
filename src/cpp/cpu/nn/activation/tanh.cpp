#include <immintrin.h>
#include <stdio.h>
#include <cmath>
using namespace std;
void tanh(float* x,int lengthN,float* activatedoutput){
    //(e^x - e^-x)/(e^x + e^-x)
    int i=0;
    __m256 zeros=_mm256_setzero_ps();
    for(;i<lengthN-7;i=i+8){
    __m256 xvector=_mm256_loadu_ps(&x[i]);
    __m256 n1=_mm256_exp_ps(xvector);
    __m256 n2=_mm256_exp_ps(_mm256_sub_ps(zeros,xvector));
    __m256 numerator=_mm256_sub_ps(n1,n2);
    __m256 denominator=_mm256_add_ps(n1,n2);
    _mm256_storeu_ps(activatedoutput+i,_mm256_div_ps(numerator,denominator));
    }
    for(;i<lengthN;i++){
        activatedoutput[i]=(exp(x[i])-exp(-x[i]))/(exp(x[i])+exp(-x[i]));
    }
}

