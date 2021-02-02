/*
* Benchmark activation / classification functions + their derivation
*/

#include <iostream>
#include <benchmark/benchmark.h>

#define SIZE_1 10
#define SIZE_2 100
#define SIZE_3 1000
#define SIZE_4 5000

#include "../include/tensorslow.h"


static void sigmoidPerf(benchmark::State& state) {
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> vec_;
	vec_.setRandom(state.range(0), 1);

	ts::WengertList<float> wList;
	ts::Tensor<float> vec(vec_, &wList);


	for(auto _ : state) {
		ts::Tensor<float> res = ts::sigmoid(vec);
		res.grad();
	}
}

BENCHMARK(sigmoidPerf)->Arg(SIZE_1)->Arg(SIZE_2)->Arg(SIZE_3)->Arg(SIZE_4);



static void reluPerf(benchmark::State& state) {
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> vec_;
	vec_.setRandom(state.range(0), 1);

	ts::WengertList<float> wList;
	ts::Tensor<float> vec(vec_, &wList);


	for(auto _ : state) {
		ts::Tensor<float> res = ts::relu(vec);
		res.grad();
	}
}

BENCHMARK(reluPerf)->Arg(SIZE_1)->Arg(SIZE_2)->Arg(SIZE_3)->Arg(SIZE_4);



BENCHMARK_MAIN();
