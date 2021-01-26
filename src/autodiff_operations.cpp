/*
* Overloaded operators and functions for the ts::Tensor class. This file only
* contains basic operations. Other more advanced functions (convolution, etc...)
* will be defined in another file.
*/

#include "../include/autodiff.hpp"


	// Overloaded arithmetic operators

template <typename T>
ts::Tensor<T> ts::operator+(const ts::Tensor<T> &x, const ts::Tensor<T> &y){
	// Element-wise sum operation

	if(
		x.wList != y.wList ||
		x.value.rows() != y.value.rows() ||
		x.value.cols() != y.value.cols()
	) {
		return ts::Tensor<T>(Eigen::Array<T, 0, 0>(), NULL);
	}


	// a = x + y
	// da / dx = 1
	// da / dy = 1

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> grad;
	grad.setOnes(x.value.rows(), x.value.cols());

	std::shared_ptr<ts::Node<T>> nodePtr (
		new ts::ElementWiseNode<T>(
			{x.value.rows(), x.value.cols()},
			grad, x.index,
			grad, y.index
		)
	);

	return ts::Tensor<T>(x.value + y.value, x.wList, nodePtr);
}



template <typename T>
ts::Tensor<T> ts::operator-(const ts::Tensor<T> &x, const ts::Tensor<T> &y){
	// Element-wise difference operation

	if(
		x.wList != y.wList ||
		x.value.rows() != y.value.rows() ||
		x.value.cols() != y.value.cols()
	) {
		return ts::Tensor<T>(Eigen::Array<T, 0, 0>(), NULL);
	}


	// a = x - y
	// da / dx = 1
	// da / dy = -1

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> grad;
	grad.setOnes(x.value.rows(), x.value.cols());

	std::shared_ptr<ts::Node<T>> nodePtr (
		new ts::ElementWiseNode<T>(
			{x.value.rows(), x.value.cols()},
			grad, x.index,
			-1 * grad, y.index
		)
	);

	return ts::Tensor<T>(x.value - y.value,x.wList, nodePtr);
}



template <typename T>
ts::Tensor<T> ts::operator*(const ts::Tensor<T> &x, const ts::Tensor<T> &y){
	// Element-wise (Hadamard) product operation

	if(
		x.wList != y.wList ||
		x.value.rows() != y.value.rows() ||
		x.value.cols() != y.value.cols()
	) {
		return ts::Tensor<T>(Eigen::Array<T, 0, 0>(), NULL);
	}


	// a = x * y
	// da / dx = y
	// da / dy = x

	std::shared_ptr<ts::Node<T>> nodePtr (
		new ts::ElementWiseNode<T>(
			{x.value.rows(), x.value.cols()},
			y.value, x.index,
			x.value, y.index
		)
	);

	return ts::Tensor<T>(x.value * y.value,x.wList, nodePtr);
}



template <typename T>
ts::Tensor<T> ts::operator/(const ts::Tensor<T> &x, const ts::Tensor<T> &y){
	// Element-wise quotient operation

	if(
		x.wList != y.wList ||
		x.value.rows() != y.value.rows() ||
		x.value.cols() != y.value.cols()
	) {
		return ts::Tensor<T>(Eigen::Array<T, 0, 0>(), NULL);
	}


	// a = x / y
	// da / dx = 1 / y
	// da / dy = -x / y^2

	std::shared_ptr<ts::Node<T>> nodePtr (
		new ts::ElementWiseNode<T>(
			{x.value.rows(), x.value.cols()},
			1.0 / y.value, x.index,
			-x.value / (y.value * y.value), y.index
		)
	);

	return ts::Tensor<T>(x.value + y.value, x.wList, nodePtr);
}



	// Matrix product

template <typename T>
ts::Tensor<T> ts::matProd(const ts::Tensor<T> &x, const ts::Tensor<T> &y) {
	// Classic matrix-matrix product

	if(x.wList != y.wList || x.value.cols() != y.value.rows()) {
		return ts::Tensor<T>(Eigen::Array<T, 0, 0>(), NULL);
	}

	// The gradient will have to be computed for a scalar
	x.wList->elementWiseOnly = false;

	// a = x.y
	// dx = y^T	(transposed)
	// dy = x^T
	// (will be used in matrix product when computing gradient)

	std::shared_ptr<ts::Node<T>> nodePtr (
		new ts::MatProdNode<T>(
			{x.value.rows(), y.value.cols()},
			y.value.matrix().transpose(), x.index,
			x.value.matrix().transpose(), y.index,
			{x.value.rows(), x.value.cols()}, {y.value.rows(), y.value.cols()}
		)
	);

	return ts::Tensor<T>( x.value.matrix() * y.value.matrix(), x.wList, nodePtr);
}



	// Activation functions

template <typename T>
ts::Tensor<T> ts::sigmoid(const ts::Tensor<T> &x) {
	// Element-wise sigmoid function

	// a = e^x / (e^x + 1) = 1 / (1 + e^-x)
	// da / dx = e^x / (e^x + 1)^2

	std::shared_ptr<ts::Node<T>> nodePtr (
		new ts::ElementWiseNode<T>(
			{x.value.rows(), x.value.cols()},
			x.value.exp() / (x.value.exp() + 1).pow(2), x.index
		)
	);

	return ts::Tensor<T>(x.value.exp() / (x.value.exp() + 1), x.wList, nodePtr);
}



template <typename T>
ts::Tensor<T> ts::relu(const ts::Tensor<T> &x) {
	// Element-wise ReLU function
	// a = max(0, x)
	// da / dx = 0 if x<= 0 ; 1 if x > 0
	// Output is then rescaled between 0 and 1

	// Apply cwise max function
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> res = x.value;
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> dx = x.value;


	#pragma omp parallel for
	for(unsigned i=0; i<res.cols(); i++) {
		for(unsigned j=0; j<res.rows(); j++) {
			res(j, i) =  (res(j,i) < 0) ? 0 : res(j, i);
			dx(j, i) =  (res(j,i) != 0) ? 1.0 : 0;
		}
	}


	// Return value
	std::shared_ptr<ts::Node<T>> nodePtr (
		new ts::ElementWiseNode<T>(
			{x.value.rows(), x.value.cols()},
			dx, x.index
		)
	);

	return ts::Tensor<T>(res, x.wList, nodePtr);

}



template <typename T>
ts::Tensor<T> ts::leakyRelu(const ts::Tensor<T> &x) {
	// Element-wise ReLU function
	// a = max(0, x)
	// da / dx = 0 if x<= 0 ; 1 if x > 0
	// Output is then rescaled between 0 and 1

	// Apply cwise max function
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> res = x.value;
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> dx = x.value;

	// Uses hardcoded 0.1 parameter for now

	#pragma omp parallel for
	for(unsigned i=0; i<res.cols(); i++) {
		for(unsigned j=0; j<res.rows(); j++) {
			res(j, i) =  (res(j,i) < 0) ? 0.1 * res(j, i) : res(j, i);
			dx(j, i) =  (res(j,i) != 0) ? 1.0 : 0.1;
		}
	}


	// Return value
	std::shared_ptr<ts::Node<T>> nodePtr (
		new ts::ElementWiseNode<T>(
			{x.value.rows(), x.value.cols()},
			dx, x.index
		)
	);

	return ts::Tensor<T>(res, x.wList, nodePtr);

}



template <typename T>
ts::Tensor<T> ts::rescale(const ts::Tensor<T> &x) {
	// Rescales tensor to 1
	// a = a / max(a)
	// da / dx = 1 / max(a)
	// Output is then rescaled between 0 and 1


	T max = x.value.maxCoeff();

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> res = x.value / max;
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> dx;
	dx.setZero(x.value.rows(), x.value.cols());
	dx = dx + max;

	// Return value
	std::shared_ptr<ts::Node<T>> nodePtr (
		new ts::ElementWiseNode<T>(
			{x.value.rows(), x.value.cols()},
			dx, x.index
		)
	);

	return ts::Tensor<T>(res, x.wList, nodePtr);

}



	// Norm functions

template <typename T>
ts::Tensor<T> ts::squaredNorm(const ts::Tensor<T> &x) {
	// Returns the square of the 2-norm / euclidean norm of a vector

	// The gradient will have to be computed for a scalar
	x.wList->elementWiseOnly = false;

	// a = norm(x)^2
	// da / dx = 2x

	std::shared_ptr<ts::Node<T>> nodePtr (
		new ts::ScalarNode<T>(
			{1, 1}, 2 * x.value.matrix(), x.index
		)
	);

	Eigen::Array<T, 1, 1> res;
	res << (T) x.value.matrix().squaredNorm();

	return ts::Tensor<T>(res, x.wList, nodePtr);
}
