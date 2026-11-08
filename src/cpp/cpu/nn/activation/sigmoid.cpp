#include <immintrin.h>
#include <stdio.h>
#include <cmath>
using namespace std;
void sigmoid(float* x,int lengthN,float* activatedoutput){
    //1/(1+e^-x)
    int i=0;
    __m256 ones=_mm256_set1_ps(1);
    __m256 zeros=_mm256_setzero_ps();
    for(;i<lengthN-7;i=i+8){
    __m256 negx=_mm256_sub_ps(zeros,_mm256_loadu_ps(&x[i]));
    __m256 result=_mm256_add_ps(ones,_mm256_exp_ps(negx));
    __m256 final=_mm256_div_ps(ones,result);
    _mm256_storeu_ps(activatedoutput+i,final);
    }
    for(;i<lengthN;i++){
        activatedoutput[i]=1/(1+exp(-x[i]));
    }
}
