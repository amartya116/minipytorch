#include <benchmark/benchmark.h>
#include <vector>
#include <cmath>
#include "linear_algebra_matrix_operations/matrixsimple.cpp"

// Benchmark for matrix addition
static void BM_MatrixAddition(benchmark::State& state) {
    int size = state.range(0);
    std::vector<std::vector<double>> A(size, std::vector<double>(size, 1.0));
    std::vector<std::vector<double>> B(size, std::vector<double>(size, 2.0));
    
    for (auto _ : state) {
        auto result = add(A, B);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetComplexityO(size * size);
}

// Benchmark for matrix multiplication
static void BM_MatrixMultiplication(benchmark::State& state) {
    int size = state.range(0);
    std::vector<std::vector<double>> A(size, std::vector<double>(size, 1.5));
    std::vector<std::vector<double>> B(size, std::vector<double>(size, 2.5));
    
    for (auto _ : state) {
        auto result = matrixmultiply(A, B);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetComplexityO(size * size * size);
}

// Benchmark for matrix transpose
static void BM_MatrixTranspose(benchmark::State& state) {
    int size = state.range(0);
    std::vector<std::vector<double>> A(size, std::vector<double>(size, 1.0));
    
    for (auto _ : state) {
        auto result = transpose(A);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetComplexityO(size * size);
}

// Benchmark for sum reduction
static void BM_MatrixSum(benchmark::State& state) {
    int size = state.range(0);
    std::vector<std::vector<double>> A(size, std::vector<double>(size, 1.5));
    
    for (auto _ : state) {
        auto result = sum1(A, 0);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetComplexityO(size * size);
}

// Benchmark for mean computation
static void BM_MatrixMean(benchmark::State& state) {
    int size = state.range(0);
    std::vector<std::vector<double>> A(size, std::vector<double>(size, 2.0));
    
    for (auto _ : state) {
        auto result = mean(A, 0);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetComplexityO(size * size);
}

// Benchmark for element-wise multiplication
static void BM_ElementwiseMultiplication(benchmark::State& state) {
    int size = state.range(0);
    std::vector<std::vector<double>> A(size, std::vector<double>(size, 1.5));
    std::vector<std::vector<double>> B(size, std::vector<double>(size, 2.5));
    
    for (auto _ : state) {
        auto result = elementwisemultiply(A, B);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetComplexityO(size * size);
}

// Register benchmarks with different matrix sizes
BENCHMARK(BM_MatrixAddition)->RangeMultiplier(2)->Range(64, 2048)->Complexity();
BENCHMARK(BM_MatrixMultiplication)->RangeMultiplier(2)->Range(64, 512)->Complexity();
BENCHMARK(BM_MatrixTranspose)->RangeMultiplier(2)->Range(64, 2048)->Complexity();
BENCHMARK(BM_MatrixSum)->RangeMultiplier(2)->Range(64, 2048)->Complexity();
BENCHMARK(BM_MatrixMean)->RangeMultiplier(2)->Range(64, 2048)->Complexity();
BENCHMARK(BM_ElementwiseMultiplication)->RangeMultiplier(2)->Range(64, 2048)->Complexity();

BENCHMARK_MAIN();
