/*
* Implementation for AD of convolution-related operations, that will typically
* be used in a CNN.
*/

#include "../include/convolution.hpp"
#include <iostream>



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



	// Convolution

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



	// Max pooling

template <typename T>
ts::PoolingNode<T>::PoolingNode(
	std::vector<long> shape,
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> xVal, int xDep,
	std::vector<unsigned> newPool
) {

	// PoolingNode specific constructor to store the size of pools
	// (this allows us to easily upscale the matrix in grad computation)

	this->rows = shape[0];
	this->cols = shape[1];

	this->values =  {xVal};	// [da/dx]
	this->dependencies =  {xDep};

	pool = newPool;
}



template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::PoolingNode<T>::incrementGradient(
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
	unsigned &j
) {

	// Used in the  ts::Tensor::grad() method. Computes the increment of a derivative
	// for a max pooling / downsample operation.


	// Upsample matrix of child derivative to match original size

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> upsample;
	upsample.setZero(this->values[j].rows(), this->values[j].cols());


	// Affect coefficients of childDerivative to upsample pools by filling each
	// pool with the corresponding value

	for(unsigned i=0; i<childDerivative.rows() / pool[0]; i++) {
		for(unsigned j=0; j<childDerivative.cols() / pool[1]; j++) {

			// Fill one pool with one value
			for(unsigned k=0; k<pool[0]; k++) {
				for(unsigned l=0; l<pool[1]; l++) {
					upsample(i * pool[0] + k, j * pool[1] + l) =
					childDerivative(i, j);
				}
			}

		}
	}


	// Compute & return element-wise product with the this->values
	// (since this->values is 0/1-flled, we will only get the coefficients in
	// the desired positions, and 0 anywhere else)
	return upsample * this->values[j];
}



template <typename T>
ts::Tensor<T> ts::maxPooling(const ts::Tensor<T> &x, std::vector<unsigned> pool) {
	// Max pooling operation : we keep only the biggest element in each pool
	// in order to reduce the size of a matrix
	// Resulting matrix is of size : (mat.x / pool.x, mat.y / pool.y)
	// (where mat.x >= pool.x and mat.y >= pool.y)

	if(pool.size() != 2) {
		return ts::Tensor<T>(Eigen::Array<T, 0, 0>(), NULL);
	}

	if(
		x.value.rows() % pool[0] != 0 ||
		x.value.cols() % pool[1] != 0
	) {
		return ts::Tensor<T>(Eigen::Array<T, 0, 0>(), NULL);
	}

	// The gradient will have to be computed for a scalar
	x.wList->elementWiseOnly = false;


	// Init result
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> res;
	res.setZero(x.value.rows() / pool[0], x.value.cols() / pool[1]);


	// Init dx
	// (dx is 1 for each max element, 0 elsewhere)
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> dx;
	dx.setZero(x.value.rows(), x.value.cols());


	unsigned xMax, yMax;
	T maxVal;


	// Compute both pooled matrix (res) and dx
	for(unsigned i=0; i<res.rows(); i++) {
		for(unsigned j=0; j<res.cols(); j++) {

			// Get index of pool's max element
			// (for now it seems the best way is to manually iterate over
			// elements)

			xMax = 0;
			yMax = 0;
			maxVal = x.value(i * pool[0], j * pool[1]);

			for(unsigned k=0; k<pool[0]; k++) {
				for(unsigned l=0; l<pool[1]; l++) {

					if(x.value(i * pool[0] + k, j * pool[1] + l) > maxVal) {
						maxVal = x.value(i * pool[0] + k, j * pool[1] + l);
						xMax = i * pool[0] + k;
						yMax = j * pool[1] + l;
					}

				}
			}

			// Assigning values for result and derivative
			res(i, j) = maxVal;
			dx(xMax, yMax) = 1.0;

		}
	}

	std::shared_ptr<ts::Node<T>> nodePtr (
		new ts::PoolingNode<T>(
			{res.rows(), res.cols()},
			dx, x.index,
			pool
		)
	);

	return ts::Tensor<T>(res, x.wList, nodePtr);
}
