#include <iostream>
#include <unordered_map>
#include <cmath>
#include <immintrin.h>
#include "../../../core/tensorimpl.hpp"

using namespace std;
using namespace mylib::tensor;

class RMSpropOptimizer {
public:
    struct State {
        float* nodeDataState = nullptr;
        bool isInitialized = false;
    };

    unordered_map<TensorImpl*, State> optimizer_states;

    ~RMSpropOptimizer() {
        for (auto& pair : optimizer_states) {
            if (pair.second.nodeDataState) {
                delete[] pair.second.nodeDataState;
            }
        }
    }

    void step(TensorImpl* thingtoptimizenode, float decayrate, float learningrate) {
        auto& s = optimizer_states[thingtoptimizenode]; 
        
        float* nodeData = static_cast<float*>(thingtoptimizenode->storage->data);
        float* nodeGradData = static_cast<float*>(thingtoptimizenode->storageforgrad->data);
        int nbytes_as_floats = thingtoptimizenode->storage->nbytes / sizeof(float);
        
        if (!s.isInitialized) {
            s.nodeDataState = new float[nbytes_as_floats]{};
            s.isInitialized = true;
        } 

        int i = 0;
        __m256 decay_simd = _mm256_set1_ps(decayrate);
        __m256 one_minus_decay_simd = _mm256_set1_ps(1.0f - decayrate);
        __m256 lr_simd = _mm256_set1_ps(learningrate);
        __m256 eps_simd = _mm256_set1_ps(1e-8f);

        for (; i + 8 <= nbytes_as_floats; i += 8) {
            __m256 data = _mm256_loadu_ps(&nodeData[i]);
            __m256 grad = _mm256_loadu_ps(&nodeGradData[i]);
            __m256 gradsquared = _mm256_mul_ps(grad, grad);
            __m256 statevalue = _mm256_loadu_ps(&s.nodeDataState[i]);

            statevalue = _mm256_add_ps(
                _mm256_mul_ps(decay_simd, statevalue),
                _mm256_mul_ps(one_minus_decay_simd, gradsquared)
            );
            _mm256_storeu_ps(&s.nodeDataState[i], statevalue);  

            __m256 datatochange = _mm256_div_ps(
                _mm256_mul_ps(lr_simd, grad),
                _mm256_add_ps(_mm256_sqrt_ps(statevalue), eps_simd)
            );
            data = _mm256_sub_ps(data, datatochange);
            _mm256_storeu_ps(&nodeData[i], data);  
        }

        for (; i < nbytes_as_floats; i++) { 
            float g2 = nodeGradData[i] * nodeGradData[i];
            s.nodeDataState[i] = decayrate * s.nodeDataState[i] + (1.0f - decayrate) * g2;
            nodeData[i] -= learningrate * nodeGradData[i] / (sqrtf(s.nodeDataState[i]) + 1e-8f);
        }
    }
};