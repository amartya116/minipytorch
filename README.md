# MiniPyTorch

A performance-oriented deep learning framework implemented from first principles, featuring C++20 backends with AVX2 SIMD optimization, automatic differentiation for backpropagation, and preliminary CUDA support.

## Architecture Overview

MiniPyTorch is structured as a multi-layer stack separating computational kernels from user-facing APIs:

```
Python Interface Layer
       |
   Tensor (Python wrapper)
       |
C++ Binding Layer (pybind11)
       |
Core Computational Engine
       ├── linear_algebra_matrix_operations/
       ├── autograd/
       ├── cpu/
       └── cuda/
```

The framework implements a traditional autodiff architecture with computational graphs that track operations for gradient computation during the backward pass.

## Core Components

### Linear Algebra & Matrix Operations

Located in `src/cpp/linear_algebra_matrix_operations/`, this module implements fundamental tensor operations:

- Element-wise operations: addition, subtraction, multiplication, division
- Unary operations: square root, square, exponential, logarithm
- Reduction operations: sum, product, mean, min, max
- Advanced decompositions: Singular Value Decomposition (SVD), eigenvalue decomposition, matrix inversion
- Linear transformations: matrix multiplication, transpose, reshape, flatten, trace

SIMD acceleration via AVX2 is applied to element-wise and reduction operations where data parallelism can be exploited. The `simdadd` kernel demonstrates SIMD-accelerated addition with explicit vector operations for float32 arrays.

### Automatic Differentiation

Located in `src/cpp/autograd/`, the autodiff system tracks computational graphs and computes gradients via backpropagation:

- Computational graph nodes track operations and their inputs
- Supported node types: `AddNode`, `MulNode`, `ActivationNode`, `LossFunctionNode`
- Gradient accumulation during backward pass
- Integration with optimizer update logic

The autograd layer currently has incomplete node implementations; header include cycles must be resolved to enable full compilation.

### CPU Backend

`src/cpp/cpu/` contains the primary computational backend for x86-64 architectures. Operations use standard C++ with compiler-level optimizations (O2/O3) and architectural flags (AVX2).

### GPU Backend

`src/cpp/cuda/` is a stub for CUDA-based operations. This module will contain kernel implementations for NVIDIA GPUs using CUDA compute kernels and cuBLAS for optimized linear algebra.

## Build System

### Requirements

- CMake 3.15+
- C++20 capable compiler (MSVC 2019+, GCC 10+, Clang 10+)
- Python 3.8+ (development headers and libs)
- pybind11 2.6+
- vcpkg for dependency management (optional but recommended)

### Compilation

The project uses CMake for cross-platform builds with platform-specific optimization flags:

```sh
mkdir -p build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

On Windows with MSVC, AVX2 is enabled via `/arch:AVX2` flag. On Unix-like systems (GCC/Clang), `-mavx2` is used. Optimization level is set to `/O2` (MSVC) or `-O3` (GCC/Clang).

### CMake Targets

- `minipytorch`: Main library target
- `matrix_ops_simple`: pybind11 module for matrix operations
- `initialization`: pybind11 module for network initialization routines

## Python Interface

The `Tensor` class in `src/pythoninterface/Tensor.py` provides the user-facing API, wrapping C++ implementations via pybind11:

```python
from pythoninterface.Tensor import Tensor

t1 = Tensor([1.0, 2.0, 3.0])
t2 = Tensor([4.0, 5.0, 6.0])

result = t1.add(t2)
transposed = t1.T
reshaped = t1.reshape((1, 3))
```

Supported operations include all basic arithmetic, linear algebra routines, and reduction functions. Operations return new Tensor objects, supporting functional programming patterns.

## Optimization Strategy

### SIMD Vectorization

Element-wise operations exploit data parallelism via AVX2 instructions, allowing 8 single-precision floats to be processed per CPU cycle. This is particularly effective for large matrices where instruction-level parallelism can hide memory latency.

### Compiler Optimizations

CMake applies aggressive optimization flags suitable for numerical compute workloads:
- Function inlining for small operator kernels
- Loop unrolling for reduction operations
- Vectorization pragmas for auto-vectorization

### Memory Layout

Matrix storage uses row-major (C-contiguous) ordering by default, compatible with standard C arrays and NumPy conventions.

## Current Development Status

### Completed

- Core C++ matrix operations with SIMD acceleration
- Python binding layer via pybind11
- Basic autograd node definitions
- CMake build infrastructure with vcpkg integration

### In Progress

- Resolving header include cycles in autograd nodes
- Implementing full backward pass gradient computation
- Optimizer integration (SGD, Adam)
- Comprehensive unit tests

### Planned

- CUDA kernel implementations
- Batch operations and dynamic shapes
- Neural network layer abstractions (Linear, Conv2D)
- Performance benchmarking against PyTorch

## Dependency Management

Dependencies are managed via vcpkg and specified in `vcpkg.json`:

- `armadillo`: Linear algebra operations
- `openblas`: High-performance BLAS library
- `eigen3`: Alternative linear algebra library
- `lapack`: Dense linear algebra routines

The `vcpkg_installed/` directory contains prebuilt binaries for the x64-windows triplet.

## Build Artifacts

Compiled binaries are located in `build/` with the following structure:

- `Debug/`: Debug builds with full symbol information
- `Release/`: Optimized Release builds
- `x64/`: Architecture-specific artifacts
- `lib.win-amd64-cpython-314/`: Python extension modules for CPython 3.14

## Known Issues

- Header include cycles prevent autograd node compilation
- CUDA backend is not yet functional
- No comprehensive test suite

## Future Directions

Performance optimization will focus on batched operations with static shape inference, reducing overhead in graph construction. GPU support requires complete CUDA kernel implementation and memory management on device. Extended coverage of neural network layers will enable end-to-end training of common architectures.
