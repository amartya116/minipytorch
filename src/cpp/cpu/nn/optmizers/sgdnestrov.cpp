#include <iostream>
#include <vector>
#include <immintrin.h>
#include "../../../core/tensorimpl.hpp"
using namespace std;
using namespace mylib::tensor;
class optmierSGDmomentum{
    public:
        struct State{
            float* nodeDataState = nullptr;
            bool isInitialized = false;
        };
        unordered_map<TensorImpl*,State> optimizer_states;

        ~optmierSGDmomentum()
        {
            for (auto& pair : optimizer_states) {
            if (pair.second.nodeDataState) {
                delete[] pair.second.nodeDataState;
            }
        }
        }
void step(float learningrate,float beta,TensorImpl* thingtoptimizenode){
    auto& s = optimizer_states[thingtoptimizenode]; 
    float* nodeData = static_cast<float*>(thingtoptimizenode->storage->data);
    float* nodeGradData = static_cast<float*>(thingtoptimizenode->storageforgrad->data);
    float* velocity = static_cast<float*>(thingtoptimizenode->storage->data);
    int nbytes_as_floats = thingtoptimizenode->storage->nbytes / sizeof(float);
      if (!s.isInitialized) {
            s.nodeDataState = new float[nbytes_as_floats]{};
            s.isInitialized = true;
        }
    __m256 betavector=_mm256_set1_ps(beta);
    __m256 learningratevector=_mm256_set1_ps(learningrate);
        for(;i<nbytes_as_floats-7;i+=8){
            __m256 velocitysimd=(epoch==1)?_mm256_setzero_ps():_mm256_loadu_ps(velocity + i);
            __m256 A=_mm256_loadu_ps(nodeData + i);
            __m256 B=_mm256_loadu_ps(nodeGradData + i);
            __m256 velocitynew=_mm256_add_ps(_mm256_mul_ps(betavector,velocitysimd),_mm256_mul_ps(learningratevector,B));   
            _mm256_storeu_ps(velocity + i, velocitynew);
            __m256 newupdatedweight=_mm256_sub_ps(A,_mm256_add_ps(_mm256_mul_ps(velocitynew,betavector),_mm256_mul_ps(learningratevector,B)));
            _mm256_storeu_ps(nodeData + i, newupdatedweight);
    }
    for(;i<nbytes_as_floats;i++){
        velocity[i]=beta*velocity[i]+learningrate*nodeGradData[i];
        nodeData[i]=nodeData[i]-(beta*velocity[i]+learningrate*nodeGradData[i]);
    }
}};