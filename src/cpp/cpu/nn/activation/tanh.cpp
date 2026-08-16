#include <immintrin.h>
#include <stdio.h>
#include <cmath>
using namespace std;
void tanh(float* x,int lengthN,float* activatedoutput){
    //1/1+(pow,x)
    int i=0;
    for(;i<lengthN-7;i=i+7){
    __m256 ones=_mm256_set1_ps(1);
    __m256 n1=_mm256_exp_ps(_mm256_div_ps(ones,_mm256_loadu_ps(&x[i])));
    __m256 n2=_mm256_exp_ps(_mm256_loadu_ps(&x[i]));
    __m256 numerator=_mm256_add_ps(n1,n2);
    __m256 denominator=_mm256_sub_ps(n1,n2);
    _mm256_storeu_ps(activatedoutput,_mm256_div_ps(numerator,denominator));
    }
    for(;i<lengthN;i++){
        activatedoutput[i]=exp(x[i])-exp(1/x[i])/exp(x[i])+exp(1/x[i]);
    }
}
 