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

	// No operation for this node (input var)
};



template <typename T>
ts::Node<T>::Node(
	std::vector<long> shape, ts::OperationType newOperationType,
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> xVal, int xDep
) {
	rows = shape[0];
	cols = shape[1];

	operationType = newOperationType;

	values =  {xVal};	// [da/dx]
	dependencies =  {xDep};
}



template <typename T>
ts::Node<T>::Node(
	std::vector<long> shape, ts::OperationType newOperationType,
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> xVal, int xDep,
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> yVal, int yDep
) {
	rows = shape[0];
	cols = shape[1];

	operationType = newOperationType;

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
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::Tensor<T>::incrementGradient(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> * childDerivative,
		ts::Node<T> * node, unsigned j
) {

	// Used in the ts::grad() method. Computes the increment of a derivative
	// depending on its operation type.

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> increment;

	switch(node->operationType) {

		case ts::ElementWise: {
			increment = node->values[j] * *childDerivative;
			break;
		}

		case ts::MatrixProduct: {
			// Make sure operands are at the correct position
			if(node->values[j].cols() == childDerivative->rows()) {
				increment =
				(node->values[j].matrix() * childDerivative->matrix()).array();
			}
			else if(node->values[j].rows() == childDerivative->cols()) {
				increment =
				(childDerivative->matrix() * node->values[j].matrix() ).array();
			}
			break;
		}

		case ts::Norm: {
			increment =
			node->values[j] * (*childDerivative)(0, 0);
			break;
		}

		default: {
			// ts::None or unhandled value : do nothing
			// (this shouldn't happen)
		}
	}

	return increment;
}



template <typename T>
ts::Gradient<T> ts::Tensor<T>::grad() {
	// Computes the gradient of this variable with respect to all the Wengert
	// list's nodes. Derivatives are stored in a vector wich size equals the
	// Wengert list's.

	// 2 possibilities :
	// - All operations are element wise, so we allow this tensor not to be a scalar
	// - Some operations change shapes of tensors, we only allow this tensor to be scalar

	// Making sure that we're not in case 2 with a non-scalar tensor
	if(!wList->elementWiseOnly && value.rows() != 1 && value.cols() != 1) {
		return ts::Gradient<T>({});
	}


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
		for(unsigned j = 0; j < node->dependencies.size(); j++) {
			derivatives[node->dependencies[j]] += incrementGradient(
				&(derivatives[i]), node, j
			);
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



template <typename T>
bool ts::Gradient<T>::isEmpty() {
	// Used to look for errors after computing a gradient
	return derivatives.size() == 0 ? true : false;
}
