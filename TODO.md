- [x] Confirm current build/run target(s) for `linear.cpp` (linear target vs minipytorch.exe)

- [ ] Fix header include errors by removing `.cpp` includes from:
  - src/cpp/autograd/autogradnodes/addnode.hpp
  - src/cpp/autograd/autogradnodes/mulnode.hpp
  - src/cpp/autograd/autogradnodes/activationnode.hpp
  - src/cpp/autograd/autogradnodes/lossfunctionnode.hpp
- [ ] Ensure headers include correct forward declarations / includes for `TensorImpl` and `Dtypes`.
- [x] Rebuild with `cmake --build build --config Release --target linear` and verify compilation (CMake target not available; MSBuild expects other targets instead)

- [ ] Run the produced executable and verify printed output matches `expected` in `linear.cpp`
- [ ] If output mismatch: inspect autograd backward for MSE + matmul/add gradients and optimizer update logic
- [ ] Update TODO progress accordingly

