#include <iostream>
using namespace std;
void gemmnaive(float* A,float* B,float* C,int M,int K,int N){
    std::fill(C, C + M*N, 0.0f);
    for(int i=0;i<M;i++){
        std::fill(C, C + M, 0.0f);
        for(int k=0;k<K;k++){
            float a = A[i*K + k]; 
            for(int j=0;j<N;j++){
                    C[i*N+j]+=a*B[k*N+j];        
        }
        } 
    }
}
int main()
{

}