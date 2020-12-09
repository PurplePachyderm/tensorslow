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
