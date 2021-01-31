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
#define SIZE_4 500

#include "../include/tensorslow.h"


// TODO Convert this benchmark to compare naive vs im2col approaches
// for multichannel convolutions.

static void naiveConv(benchmark::State& state) {

	// Prepare data

	ts::WengertList<float> wList;

	std::vector<std::vector<ts::Tensor<float>>> kernels = {};

	// Gen kernels
	for(unsigned i=0; i<32; i++) {
		kernels.push_back({});
		for(unsigned j=0; j<3; j++) {
			Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> ker;
			ker.setRandom(KERNEL_SIZE, KERNEL_SIZE);
			kernels[i].push_back(
				ts::Tensor<float>(ker, &wList)
			);
		}
	}

	// Gen input matrices
	std::vector<ts::Tensor<float>> mat = {};
	for(unsigned i=0; i<3; i++) {
		Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> mat_;
		mat_.setRandom(state.range(0), state.range(0));

		mat.push_back(
			ts::Tensor<float>(mat_, &wList)
		);
	}

	// Results array (empty for now)
	std::vector<ts::Tensor<float>> res = {};


	// Compute multilayer conv
	for(auto _ : state) {

		for(unsigned i=0; i<32; i++) {
			// Push a zero filled matrix for each output channel
			res.push_back(
				ts::Tensor<float>(
					Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>()
					.setZero(
						state.range(0) - KERNEL_SIZE + 1, state.range(0) - KERNEL_SIZE + 1
					),
					&wList
				)
			);

			for(unsigned j=0; j<3; j++) {
				res[i] = res[i] + ts::convolution(mat[j], kernels[i][j]);
			}
		}

	}
}

BENCHMARK(naiveConv)->Arg(SIZE_1)->Arg(SIZE_2)->Arg(SIZE_3)->Arg(SIZE_4);



static void im2col(benchmark::State& state) {

	ts::WengertList<float> wList;

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> kernel_;
	kernel_.setRandom(32, 27);
	ts::Tensor<float> kernel = ts::Tensor<float>(kernel_, &wList);

	// Gen input matrices
	std::vector<ts::Tensor<float>> mat = {};
	for(unsigned i=0; i<3; i++) {
		Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> mat_;
		mat_.setRandom(state.range(0), state.range(0));

		mat.push_back(
			ts::Tensor<float>(mat_, &wList)
		);
	}


	for(auto _ : state) {

		ts::Tensor<float> im2colMat = ts::im2col(mat, {KERNEL_SIZE, KERNEL_SIZE});
		ts::Tensor<float> res = ts::matProd(kernel, im2colMat);
		std::vector<ts::Tensor<float>> resVec = ts::col2im(
			res,
			{
				(unsigned) (state.range(0) - KERNEL_SIZE + 1),
				(unsigned) (state.range(0) - KERNEL_SIZE + 1)
			}
		);

	}
}

BENCHMARK(im2col)->Arg(SIZE_1)->Arg(SIZE_2)->Arg(SIZE_3)->Arg(SIZE_4);



BENCHMARK_MAIN();
