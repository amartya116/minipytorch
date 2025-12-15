cmake_minimum_required(VERSION 3.15)
project(minipytorch)

set(CMAKE_CXX_STANDARD 17)

# Find Python
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Find pybind11
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# Add the pybind11 module
pybind11_add_module(example linear_algebra_matrix_operations/vectoradd.cpp)

# Include directories if needed
target_include_directories(example PRIVATE ${Python_INCLUDE_DIRS})
