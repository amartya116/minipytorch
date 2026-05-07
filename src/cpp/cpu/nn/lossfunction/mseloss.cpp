#include <iostream>
#include <vector>
#include <immintrin.h>
using namespace std;

float mseloss(float* pred,float* actual,int size){
    int i=0;
    float loss=0;
    float temp[8];

    for(;i+ 7<size;i=i+8){
        __m256 A=_mm256_loadu_ps(&pred[i]);
        __m256 B=_mm256_loadu_ps(&actual[i]);
        __m256 intc = _mm256_sub_ps(A,B);
        __m256 C=_mm256_mul_ps(intc,intc);
        _mm256_storeu_ps(temp,C);
        loss+=temp[0]+temp[1]+temp[2]+temp[3]+temp[4]+temp[5]+temp[6]+temp[7];
    }
    for(;i<size;i++){
        loss+=(pred[i]-actual[i])*(pred[i]-actual[i]);
    }
    return loss/size;
    
}
