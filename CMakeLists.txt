cmake_minimum_required(VERSION 3.15)
project(minipytorch)

set(CMAKE_CXX_STANDARD 20)

include_directories(src/cpp)

# Enable SIMD and CPU optimizations
if(MSVC)
    # Visual Studio compiler
    add_compile_options(/arch:AVX2)  # AVX2 SIMD support
    add_compile_options(/O2)          # Optimization level
else()
    # GCC/Clang
    add_compile_options(-mavx2)       # AVX2 SIMD support
    add_compile_options(-O3)          # Optimization level
endif()

# Find Python
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Find pybind11
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# Add the pybind11 module
pybind11_add_module(matrix_ops_simple src/cpp/linear_algebra_matrix_operations/matrixsimple.cpp)
pybind11_add_module(initialization src/cpp/autograd/Initialization/initneu.cpp)

# Include directories if needed
target_include_directories(matrix_ops_simple PRIVATE ${Python_INCLUDE_DIRS})