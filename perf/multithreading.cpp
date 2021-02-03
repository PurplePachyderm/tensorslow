/*
* Benchmark for Eigen parallelization  with different numbers of threads.
* Results may vary depending on your hardware, especially if you use more
* threads than your number of physical cores.
*/


#include <iostream>
#include <benchmark/benchmark.h>
#include <omp.h>

#include "../include/tensorslow.h"


// Change these values to modify model size
#define INPUT_SIZE 1000
#define LAYER_SIZE 1000

#define NTHREADS_1 1
#define NTHREADS_2 2
#define NTHREADS_3 4
#define NTHREADS_4 8

#define NLAYERS_1 16
#define NLAYERS_2 32
#define NLAYERS_3 64




// MLP

static void lightMLP(benchmark::State& state) {

	// Multithreaded should be similar to single threaded, probably because
	// of parallelism overhead (though it may differ by machine).

	omp_set_num_threads(state.range(0));

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

BENCHMARK(lightMLP)->Arg(NTHREADS_1)->Arg(NTHREADS_2)->Arg(NTHREADS_3)->Arg(NTHREADS_4);



// CNN

static void heavyCnn(benchmark::State& state) {

	// Since this model is way longe to compute, multi-threaded performances
	// should start getting sligtly better.

	omp_set_num_threads(state.range(0));

	ts::ConvolutionalNetwork<float> model(
		// Input
		{96, 32},

		// Number of channels for input (3 for RGB)
		ts::ChannelSplit::SPLIT_HOR, 3,

		// Convolution / pooling
		{{3, 3, 128}, {5, 5, 128}},
		{{0,0}, {2, 2}},

		// Dense layers (with output vector & not including first layer)
		{256, 128, 10}
	);
	model.toggleGlobalOptimize(true);

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> input_;
	input_.setRandom(96, 32);


	for(auto _ : state) {
		ts::Tensor<float> input = ts::Tensor<float>(input_, &(model.wList));
		ts::squaredNorm(model.compute(input)).grad();
		model.wList.reset();
	}

}

BENCHMARK(heavyCnn)->Arg(NTHREADS_1)->Arg(NTHREADS_2)->Arg(NTHREADS_3)->Arg(NTHREADS_4);


// MAIN

BENCHMARK_MAIN();
