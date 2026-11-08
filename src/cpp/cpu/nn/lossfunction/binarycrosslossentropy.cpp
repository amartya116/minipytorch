#include <iostream>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

using namespace std;
float BCEloss(float* inputfromlastnode,float* actual,int size){
    int i=0;
    float loss=0.0f;
    float eps=1e-7f;    // keep p away from 0 and 1 so log never sees 0
    __m256 epsvector=_mm256_set1_ps(eps);
    __m256 onemepsvector=_mm256_set1_ps(1.0f-eps);
    for(;i<size-7;i=i+8){
        float temp[8];
        __m256 inputa=_mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(&inputfromlastnode[i]),epsvector),onemepsvector);
        __m256 actualsimd=_mm256_loadu_ps(&actual[i]);
        // y*log(p) + (1-y)*log(1-p)
        __m256 result=_mm256_add_ps(_mm256_mul_ps(actualsimd,_mm256_log_ps(inputa)),_mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1),actualsimd),_mm256_log_ps(_mm256_sub_ps(_mm256_set1_ps(1),inputa))));
        _mm256_storeu_ps(temp,result);
        loss-=temp[0]+temp[1]+temp[2]+temp[3]+temp[4]+temp[5]+temp[6]+temp[7];
}
    for(;i<size;i++){
        float p=max(eps,min(inputfromlastnode[i],1.0f-eps));
        loss-=actual[i]*log(p)+(1-actual[i])*log(1-p);
    }
    return loss/size;
}
