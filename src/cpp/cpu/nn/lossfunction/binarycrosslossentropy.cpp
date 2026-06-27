#include <iostream.h>
#include <immintrin.h>       

using namespace std;
float BCEloss(float* inputfromlastnode,float* actual,int size){
    int i=0;
    float loss;
    for(;i<size-7;i=i+8){
        float temp[8];
        __m256 inputa=_mm256_loadu_ps(&inputfromlastnode[i]);
        __m256 actualsimd=_mm256_loadu_ps(&actual[i]);
        __m256 result=_mm256_add_ps(_mm256_mul_ps(actualsimd,_mm256_logb_ps(inputa)),_mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1),actualsimd),_mm256_logb_ps(_mm256_sub_ps(_mm256_set1_ps(1),inputa))));
        _mm256_storeu_ps(temp,result);
        loss+=abs(temp[0])+abs(temp[1])+abs(temp[2])+abs(temp[3])+abs(temp[4])+abs(temp[5])+abs(temp[6])+abs(temp[7]);
}
    for(;i<size;i++){
        loss+=abs(log(inputfromlastnode[i])*actual[i]-log(1-inputfromlastnode[i])*(1-actual[i]));
    }
    return loss/size;
}