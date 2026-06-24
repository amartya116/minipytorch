#include <iostream>
#include <immintrin.h>
#include "../../../core/tensorimpl.hpp"
using namespace std;
using namespace mylib::tensor;

void RMSprop(TensorImpl* thingtoptimizenode,float decayrate,float learningrate){
    struct State{
        TensorImpl* Tensorofcurrentnode;
        float* nodeDataState;
        bool insinitilzed=false;
    };
    unordered_map<TensorImpl*,State> state_;
    auto s=state_[thingtoptimizenode];
    int i=0;
    float* nodeData = static_cast<float*>(thingtoptimizenode->storage->data);
    float* nodeGradData = static_cast<float*>(thingtoptimizenode->storageforgrad->data);
    int nbytes_as_floats = thingtoptimizenode->storage->nbytes / sizeof(float);
    if (!s.insinitilzed) {
                s.nodeDataState = new float[nbytes_as_floats]{}; ;  // first time seeing this param: make v = 0, same shape
                s.insinitilzed = true;
            } 
  for (; i + 8 <= nbytes_as_floats; i += 8) {
    __m256 data = _mm256_loadu_ps(&nodeData[i]);
    __m256 grad = _mm256_loadu_ps(&nodeGradData[i]);
    __m256 gradsquared = _mm256_mul_ps(grad, grad);
    __m256 betasimd = _mm256_set1_ps(decayrate);
    __m256 statevalue = _mm256_loadu_ps(&s.nodeDataState[i]);

    statevalue = _mm256_add_ps(
        _mm256_mul_ps(betasimd, statevalue),
        _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), betasimd), gradsquared)
    );
    _mm256_storeu_ps(&s.nodeDataState[i], statevalue);  

    __m256 datatochange = _mm256_div_ps(
        _mm256_mul_ps(_mm256_set1_ps(learningrate), grad),
        _mm256_add_ps(_mm256_sqrt_ps(statevalue), _mm256_set1_ps(1e-8f))
    );
    data = _mm256_sub_ps(data, datatochange);
    _mm256_storeu_ps(&nodeData[i], data);  
}
for (; i < nbytes_as_floats; i++) { 
    float g2 = nodeGradData[i] * nodeGradData[i];
    s.nodeDataState[i] = decayrate * s.nodeDataState[i] + (1.0f - decayrate) * g2;
    nodeData[i] -= learningrate * nodeGradData[i] / (sqrtf(s.nodeDataState[i]) + 1e-8f);
}