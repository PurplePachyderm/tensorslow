/*
* Implementation for AD of convolution-related operations, that will typically
* be used in a CNN.
*/

#include "../include/convolution.hpp"



// Convolution operation on Eigen arrays
// (will be reused in ts::Tensor convolutions)
template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::convArray(
	const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &mat,
	const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &ker
) {

	// Make sure kernel is smaller
	if(
		mat.rows() < ker.rows() ||
		mat.cols() < ker.cols()
	) {
		return Eigen::Array<T, 0, 0>();
	}


	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> res;
	res.resize(mat.rows() - ker.rows() + 1, mat.cols() - ker.cols() + 1);

	for(unsigned i=0; i<mat.rows() - ker.rows() + 1; i++) {
		for(unsigned j=0; j<mat.cols() - ker.cols() + 1; j++) {
			// Compute one element of feature map
			res(i, j) =
			(mat.block(i, j, ker.rows(), ker.cols()) * ker).sum();
		}
	}

	return res;
}



	// Convolution node

template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::ConvolutionNode<T>::incrementGradient(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
		unsigned &j
) {

	// Used in the  ts::Tensor::grad() method. Computes the increment of a derivative
	// for a convolution operation.

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> increment;

	// Matrices are already prepared at this stage, so we only need to put the
	// operands in the correct order for convolution.

	if(
		childDerivative.rows() > this->values[j].rows() &&
		childDerivative.rows() > this->values[j].cols()
	) {
		increment = ts::convArray(childDerivative, this->values[j]);
	} else {
		increment = ts::convArray(this->values[j], childDerivative);
	}

	return increment;
}



	// Convolution operation

template <typename T>
ts::Tensor<T> ts::convolution(const ts::Tensor<T> &mat, const ts::Tensor<T> &ker) {
	// Convolution operation
	// Resulting matrix is of size : (mat.x - ker.x + 1, mat.y - ker.y + 1)
	// (where mat.x >= ker.x and mat.y >= mat.y)

	if(
		mat.wList != ker.wList ||
		mat.value.rows() < ker.value.rows() ||
		mat.value.cols() < ker.value.cols()
	) {
		return ts::Tensor<T>(Eigen::Array<T, 0, 0>(), NULL);
	}

	// The gradient will have to be computed for a scalar
	mat.wList->elementWiseOnly = false;


	// res = conv(mat, ker)
	// dMat = ker.rotate(180) + padding
	// dKer = mat
	// (will be used in another convolution for gradient)

	// Compute res
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> res = ts::convArray(
		mat.value, ker.value
	);

	// Init dMat matrix (for matrix partial derivative)
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> dMat;
	dMat.resize(
		2 * res.rows() + ker.value.rows() - 2,
		2 * res.cols() + ker.value.cols() - 2
	);

	dMat.block(
		res.rows() - 1,
		res.cols() - 1,
		ker.value.rows(), ker.value.cols()
	) = ker.value.rowwise().reverse().colwise().reverse();

	std::shared_ptr<ts::Node<T>> nodePtr (
		new ts::ConvolutionNode<T>(
			{res.rows(), res.cols()},
			dMat, mat.index,
			mat.value, ker.index
		)
	);

	return ts::Tensor<T>(res, mat.wList, nodePtr);
}
