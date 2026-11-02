#include <benchmark/benchmark.h>
#include <vector>
#include "linear_algebra_matrix_operations/naivegemm.cpp"

// Small edge-case shape: M=7 (6+1 row tail), N=17 (16+1 col tail), K=5
static void GemmBench(benchmark::State& state) {
	// A is M x K = 7 x 5, row-major
	float A[7*5] = {
		1, 2, 3, 4, 5,
		2, 0, 1, 3, 1,
		0, 1, 0, 2, 4,
		3, 1, 2, 0, 1,
		1, 1, 1, 1, 1,
		4, 0, 2, 1, 3,
		2, 3, 1, 0, 2
	};
	// B is K x N = 5 x 17, row-major
	float B[5*17] = {
		1, 0, 2, 1, 3, 0, 1, 2, 1, 0, 1, 2, 3, 1, 0, 2, 1,
		0, 1, 1, 2, 0, 1, 3, 1, 0, 2, 1, 0, 1, 2, 1, 0, 3,
		2, 1, 0, 1, 1, 2, 0, 1, 3, 1, 0, 1, 2, 0, 1, 1, 2,
		1, 2, 1, 0, 2, 1, 1, 0, 1, 2, 3, 1, 0, 1, 2, 1, 0,
		0, 1, 2, 1, 0, 3, 1, 2, 0, 1, 1, 2, 1, 0, 1, 3, 1
	};
	float C[7*17];
	int N=17, K=5, M=7;
	for (auto _ : state) {
		gemmmatmul(N,M,K,A,B,C);
		benchmark::DoNotOptimize(C);
		benchmark::ClobberMemory();
	}
}
BENCHMARK(GemmBench);

// Square sizes, reports GFLOP/s (2*M*N*K flops per call)
static void GemmSquare(benchmark::State& state) {
	int n = state.range(0);
	std::vector<float> A(n*n, 1.0f), B(n*n, 0.5f), C(n*n);
	for (auto _ : state) {
		gemmmatmul(n,n,n,A.data(),B.data(),C.data());
		benchmark::DoNotOptimize(C.data());
		benchmark::ClobberMemory();
	}
	state.counters["GFLOPS"] = benchmark::Counter(
		2.0*n*n*n, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}
BENCHMARK(GemmSquare)->Arg(128)->Arg(256)->Arg(512)->Arg(1000)->Arg(1024)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
