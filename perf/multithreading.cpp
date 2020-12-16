/*
* Benchmark for Eigen parallelization  with different numbers of threads.
* Results may vary depending on your hardware, especially if you use more
* threads than your number of physical cores.
*/

#include <iostream>
#include <benchmark/benchmark.h>

#include "../include/tensorslow.h"

// Change these values to modify model size
#define INPUT_SIZE 2000
#define LAYER_SIZE 1000



// 1 THREAD

static void mlp1Thread(benchmark::State& state) {

	omp_set_num_threads(1);

	ts::MultiLayerPerceptron<float> model(INPUT_SIZE, {LAYER_SIZE, LAYER_SIZE});
	model.toggleGlobalOptimize(true);

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> input_;
	input_.setRandom(INPUT_SIZE, 1);


	for(auto _ : state) {
		ts::Tensor<float> input = ts::Tensor<float>(input_, &(model.wList));
		ts::squaredNorm(model.compute(input)).grad();
		model.wList.reset();
	}

}

BENCHMARK(mlp1Thread);



// 2 THREADS

static void mlp2Threads(benchmark::State& state) {

	omp_set_num_threads(2);

	ts::MultiLayerPerceptron<float> model(INPUT_SIZE, {LAYER_SIZE, LAYER_SIZE});
	model.toggleGlobalOptimize(true);

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> input_;
	input_.setRandom(INPUT_SIZE, 1);


	for(auto _ : state) {
		ts::Tensor<float> input = ts::Tensor<float>(input_, &(model.wList));
		ts::squaredNorm(model.compute(input)).grad();
		model.wList.reset();
	}

}

BENCHMARK(mlp2Threads);



// 4 THREADS

static void mlp4Threads(benchmark::State& state) {

	omp_set_num_threads(4);

	ts::MultiLayerPerceptron<float> model(INPUT_SIZE, {LAYER_SIZE, LAYER_SIZE});
	model.toggleGlobalOptimize(true);

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> input_;
	input_.setRandom(INPUT_SIZE, 1);


	for(auto _ : state) {
		ts::Tensor<float> input = ts::Tensor<float>(input_, &(model.wList));
		ts::squaredNorm(model.compute(input)).grad();
		model.wList.reset();
	}

}

BENCHMARK(mlp4Threads);



// 8 THREADS

static void mlp8Threads(benchmark::State& state) {

	omp_set_num_threads(8);

	ts::MultiLayerPerceptron<float> model(INPUT_SIZE, {LAYER_SIZE, LAYER_SIZE});
	model.toggleGlobalOptimize(true);

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> input_;
	input_.setRandom(INPUT_SIZE, 1);


	for(auto _ : state) {
		ts::Tensor<float> input = ts::Tensor<float>(input_, &(model.wList));
		ts::squaredNorm(model.compute(input)).grad();
		model.wList.reset();
	}

}

BENCHMARK(mlp8Threads);



// MAIN

BENCHMARK_MAIN();
