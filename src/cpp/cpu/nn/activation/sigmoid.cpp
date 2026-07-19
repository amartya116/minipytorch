#include <immintrin.h>
#include <stdio.h>
#include <cmath>
using namespace std;
void sigmoid(float* x,int lengthN,float* activatedoutput){
    //1/1+(pow,x)
    int i=0;
    for(;i<lengthN-7;i=i+7){
    __m256 ones=_mm256_set1_ps(1);
    __m256 result=_mm256_exp_ps(_mm256_loadu_ps(&x[i]));
    __m256 final=_mm256_div_ps(ones,result);
    _mm256_storeu_ps(activatedoutput,final);
    }
    for(;i<lengthN;i++){
        activatedoutput[i]=1/exp(x[i]);
    }
}
