#include <iostream>
#include <immintrin.h>
#include "src\cpp\core\tensorimpl.hpp"
#include "src\cpp\cpu\nn\activation\relu.h"
using namespace std;
using namespace mylib::tensor;
TensorImpl* lossfunctionnode(TensorImpl* input1, const vector<float>& target,enum Dtypes dtype)
{
    int resultdatasize = input1->storage->nbytes;
    auto resultStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
    auto resultGradStorage = make_shared<Storage>(resultdatasize, input1->storage->dtype, input1->storage->device);
    TensorImpl *lossnodeMSE = new TensorImpl(resultStorage, resultGradStorage, input1->shape, input1->strides, 0,
        [target](TensorImpl* NodeInput){ MSElossfunctionnodebackward(NodeInput, target); }, "MSElossfunctionnodebackward", 0, true, {input1});
    return lossnodeMSE;
}
//MSE node Autodiff
void MSElossfunctionnodebackward(TensorImpl* NodeInput, const vector<float>& target){
    int i=0;
    float* nodeGradData = static_cast<float*>(NodeInput->storageforgrad->data);
    TensorImpl* pred = NodeInput->parents[0];
    float* nodeprevData = static_cast<float*>(pred->storage->data);
    int N = pred->storage->nbytes;
   // loss->grad[0] should be 1
    float dL_dloss = nodeGradData[0];
    

    for (; i < N; i++) {
        float y_pred = nodeprevData[i];
        float y_true = target[i];
        float dloss_dpred = (2.0f / N) * (y_pred - y_true);  // d(MSE)/dpred
        nodeGradData[i] += dL_dloss * dloss_dpred;
    }
}