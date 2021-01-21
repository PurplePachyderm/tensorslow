/*
* Implementation for AD of convolution-related operations, that will typically
* be used in a CNN. This also includes corresponding node types.
*/

#pragma once

#include "autodiff.hpp"
#include "utils.hpp"

#include <vector>
#include <memory>
#include <iostream>

#include <Eigen/Dense>

// Enum for channel splitting directions in CNN
// (declared outside for now because scoped enum declarationb seems
// impossible)
enum class ts::ChannelSplit : int {
	NOSPLIT,
	SPLIT_HOR,	// Splits lines
	SPLIT_VERT	// Splits columns
};


namespace ts {

	template <typename T>
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> convArray(
		const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &mat,
		const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &ker
	);
	template <typename T>
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> im2conv(
		const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &mat,
		const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &ker
	);

	template <typename T> class ConvolutionNode;
	template <typename T>
	ts::Tensor<T> convolution(const ts::Tensor<T> &mat, const ts::Tensor<T> &ker);

	template <typename T> class SplitNode;
	template <typename T>
	std::vector<ts::Tensor<T>> split(
		const ts::Tensor<T> &x,
		ChannelSplit channelSplit,
		unsigned nInputChannels
	);

	template <typename T> class PoolingNode;
	template <typename T>
	ts::Tensor<T> maxPooling(const ts::Tensor<T> &x, std::vector<unsigned> pool);

	template <typename T> class VertCatNode;
	template <typename T>
	ts::Tensor<T> vertCat(const std::vector<ts::Tensor<T>> &x);

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



	// ts::SplitNode

template <typename T>
class ts::SplitNode : public ts::Node<T> {
private:
	using ts::Node<T>::Node;

	SplitNode(
		std::vector<long> shape,
		int xDep,
		std::vector<long> originalShape,
		ChannelSplit newSplitDirection,
		unsigned newPosition
	);

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> incrementGradient(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
			unsigned &j
	);

	long originalRows, originalCols;
	ChannelSplit splitDirection;
	unsigned position;

	friend std::vector<ts::Tensor<T>> ts::split<>(
		const ts::Tensor<T> &x,
		ChannelSplit channelSplit,
		unsigned nInputChannels
	);
};



	// ts::VertCatNode

template <typename T>
class ts::VertCatNode : public ts::Node<T> {
private:
	using ts::Node<T>::Node;

	// This node can have n parents !
	VertCatNode(
		std::vector<long> shape,
		std::vector<int> newDependencies,
		std::vector<long> newHeights
	);

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> incrementGradient(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
			unsigned &j
	);

	std::vector<long> heights = {};

	friend ts::Tensor<T> ts::vertCat<>(const std::vector<ts::Tensor<T>> &x);
};



	// ts::FlatteningNode

template <typename T>
class ts::FlatteningNode : public ts::Node<T> {
private:
	using ts::Node<T>::Node;

	FlatteningNode(
		std::vector<long> shape,
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> xVal, int xDep,
		std::vector<long> newSize
	);

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> incrementGradient(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
			unsigned &j
	);

	std::vector<long> size = {};

	friend ts::Tensor<T> ts::flattening<>(const ts::Tensor<T> &x);
};
