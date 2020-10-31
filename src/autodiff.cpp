/*
* General automatic differentiation engine based on a Wengert list
* implementation. Reverse mode only.
*/

#include "../include/autodiff.hpp"


	// ts::Node

template <typename T>
ts::Node<T>::Node(std::vector<long> shape) {
	rows = shape[0];
	cols = shape[1];
};



template <typename T>
ts::Node<T>::Node(
	std::vector<long> shape,
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> xVal, int xDep
) {
	rows = shape[0];
	cols = shape[1];

	values =  {xVal};	// [da/dx]
	dependencies =  {xDep};
}



template <typename T>
ts::Node<T>::Node(
	std::vector<long> shape,
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> xVal, int xDep,
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> yVal, int yDep
) {
	rows = shape[0];
	cols = shape[1];

	values =  {xVal, yVal};	// [da/dx, da/dy]
	dependencies =  {xDep, yDep};
}



	// ts::WengertList

template <typename T>
int ts::WengertList<T>::size() {
	return nodes.size();
}



	// ts::Tensor

template <typename T>
ts::Tensor<T>::Tensor(
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> newValue,
	ts::WengertList<T> * newWList
) {
	value = newValue;

	wList = newWList;

	if(wList != NULL) {
		// Add new Tensor to the Wengert list
		index = wList->nodes.size();
		// Node without dependencies (input var)
		wList->nodes.push_back(ts::Node<T>({newValue.rows(), newValue.cols()}));
	} else {
		index = -1;
	}
}



template <typename T>
ts::Tensor<T>::Tensor(
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> newValue,
	ts::WengertList<T> * newWList, ts::Node<T> node
) {
	value = newValue;

	wList = newWList;

	if(wList != NULL) {
		// Add new Tensor to the Wengert list
		index = wList->nodes.size();
		wList->nodes.push_back(node);	// This node can contain dependencies & values
	} else {
		index = -1;
	}
}



// Helper function to create new instances without syntax template
template <typename T>
ts::Tensor<T> ts::NewTensor(
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> newValue,
	ts::WengertList<T> * newWList
) {
	return ts::Tensor<T>(newValue, newWList);
}



template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::Tensor<T>::getValue() {
	return value;
}



template <typename T>
ts::Gradient<T> ts::Tensor<T>::grad() {
	// Computes the gradient of all Wengert list's nodes with respect to this
	// variable. Derivatives are stored in a vector wich size equals the
	// Wengert list's.

	std::vector< Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> > derivatives(
		wList->nodes.size(),
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>()
	);

	// Initialize all gradients with correct size zero-filled arrays
	for(unsigned i = 0; i < derivatives.size(); i++) {
		derivatives[i].setZero(wList->nodes[i].rows, wList->nodes[i].cols);
	}

	// Initialize gradient of self with respect to itself
	derivatives[index].fill(1.0);


	// Iterate over the Wengert list backwards
	for (unsigned i = wList->nodes.size(); i-- > 0; ) {
		ts::Node<T> * node = &(wList->nodes[i]);

		// Increment parent nodes
		// WARNING Only works on element-wise operations
		// (derivatives[i] might not have the same shape otherwise)
		for(unsigned j = 0; j < node->dependencies.size(); j++) {
			derivatives[node->dependencies[j]] += node->values[j] * derivatives[i];
		}
	}

	return ts::Gradient<T>(derivatives);
}



	// ts::Gradient

template <typename T>
ts::Gradient<T>::Gradient(
	std::vector< Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> > newDerivatives
) {
	derivatives = newDerivatives;
}



template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::Gradient<T>::getValue(ts::Tensor<T> a) {
	return derivatives[a.index];
}



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
	// a / dy = 1

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> grad;
	grad.setOnes(x.value.rows(), x.value.cols());

	return ts::Tensor<T>(
		x.value + y.value,
		x.wList,
		ts::Node<T>(
			{x.value.rows(), x.value.cols()},
			grad, x.index,
			grad, y.index
		)
	);
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
	// a / dy = -1

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> grad;
	grad.setOnes(x.value.rows(), x.value.cols());

	return ts::Tensor<T>(
		x.value - y.value,
		x.wList,
		ts::Node<T>(
			{x.value.rows(), x.value.cols()},
			grad, x.index,
			-1 * grad, y.index
		)
	);
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
	// a / dy = x

	return ts::Tensor<T>(
		x.value * y.value,
		x.wList,
		ts::Node<T>(
			{x.value.rows(), x.value.cols()},
			y.value, x.index,
			x.value, y.index
		)
	);
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
	// a / dy = -x / y^2

	return ts::Tensor<T>(
		x.value + y.value,
		x.wList,
		ts::Node<T>(
			{x.value.rows(), x.value.cols()},
			1.0 / y.value, x.index,
			-x.value / (y.value * y.value), y.index
		)
	);
}
