/*
* Benchmark comparing classic and im2col approaches of convolution operations
* (only tests primitive operations on Eigen::Arrays)
*/

#include <iostream>
#include <benchmark/benchmark.h>

#define KERNEL_SIZE 3
#define SIZE_1 10
#define SIZE_2 50
#define SIZE_3 250
#define SIZE_4 1000

#include "../include/tensorslow.h"


// TODO Convert this benchmark to compare naive vs im2col approaches
// for multichannel convolutions.

static void convArrayPerf(benchmark::State& state) {
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> mat;
	mat.setRandom(state.range(0), state.range(0));

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> ker;
	ker.setRandom(KERNEL_SIZE, KERNEL_SIZE);

	for(auto _ : state) {
		ts::convArray(mat, ker);
	}
}

BENCHMARK(convArrayPerf)->Arg(SIZE_1)->Arg(SIZE_2)->Arg(SIZE_3)->Arg(SIZE_4);


BENCHMARK_MAIN();
