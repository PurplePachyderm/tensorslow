/*
* Implementation for AD of convolution-related operations, that will typically
* be used in a CNN. This also includes corresponding node types.
*/

#pragma once

#include "autodiff.hpp"
#include "utils.hpp"

#include <vector>
#include <memory>

#include <Eigen/Dense>



namespace ts {

	template <typename T>
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> convArray(
		const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &mat,
		const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &ker
	);

	template <typename T> class ConvolutionNode;
	template <typename T>
	ts::Tensor<T> convolution(const ts::Tensor<T> &mat, const ts::Tensor<T> &ker);

	template <typename T> class PoolingNode;
	template <typename T>
	ts::Tensor<T> maxPooling(const ts::Tensor<T> &x, std::vector<unsigned> pool);

	template <typename T> class FlatteningNode;
	template <typename T>
	ts::Tensor<T> flattening(const ts::Tensor<T> &x);
}



	// ts::ConvolutionNode

template <typename T>
class ts::ConvolutionNode : public ts::Node<T> {
private:
	using ts::Node<T>::Node;

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> incrementGradient(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
			unsigned &j
	);
};



	// ts::PoolingNode

template <typename T>
class ts::PoolingNode : public ts::Node<T> {
private:
	using ts::Node<T>::Node;

	PoolingNode(
		std::vector<long> shape,
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> xVal, int xDep,
		std::vector<unsigned> newPool
	);

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> incrementGradient(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
			unsigned &j
	);

	std::vector<unsigned> pool = {};

	friend ts::Tensor<T> ts::maxPooling<>(
		const ts::Tensor<T> &x, std::vector<unsigned> pool
	);
};



	// ts::FlatteningNode

template <typename T>
class ts::FlatteningNode : public ts::Node<T> {
private:
	using ts::Node<T>::Node;

	FlatteningNode(
		std::vector<long> shape,
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> xVal, int xDep,
		std::vector<unsigned> newSize
	);

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> incrementGradient(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
			unsigned &j
	);

	std::vector<unsigned> size = {};

	friend ts::Tensor<T> flattening<>(const ts::Tensor<T> &x);
};
