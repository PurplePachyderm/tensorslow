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



template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::InputNode<T>::incrementGradient(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
		unsigned &j
) {

	// Used in the  ts::Tensor::grad() method. For an input node, this
	// function should never be called

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> increment;
	return increment;
}



template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::ElementWiseNode<T>::incrementGradient(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
		unsigned &j
) {

	// Used in the  ts::Tensor::grad() method. Computes the increment of a derivative
	// for an element-wise operation

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> increment;
	increment = this->values[j] * childDerivative;

	return increment;
}



template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::MatProdNode<T>::incrementGradient(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
		unsigned &j
) {

	// Used in the  ts::Tensor::grad() method. Computes the increment of a derivative
	// for a matrix-matrix product.

	// BUG If two matrices of same dimensions are multiplied, for one of them,
	// the wrong operand will be selected, resulting in a matrix of the wrong
	// size.

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> increment;

	// Make sure operands are at the correct position
	if(this->values[j].cols() == childDerivative.rows()) {
		increment = (this->values[j].matrix() * childDerivative.matrix()).array();
	}
	else if(this->values[j].rows() == childDerivative.cols()) {
		increment = (childDerivative.matrix() * this->values[j].matrix() ).array();
	}

	return increment;
}



template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::ScalarNode<T>::incrementGradient(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
		unsigned &j
) {

	// Used in the ts::Tensor::grad() method. Computes the increment of a derivative
	// for a tensor to scalar operation.

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> increment;
	increment = this->values[j] * childDerivative(0, 0);

	return increment;
}



	// ts::WengertList

template <typename T>
int ts::WengertList<T>::size() {
	return nodes.size();
}



template <typename T>
int ts::WengertList<T>::reset() {
	// Used to remove all nodes but the input nodes, so the input tensors can
	// be reused in new computations. Returns the new size of the list.

	// First pass : remove non optimizable variables
	for(unsigned i = nodes.size(); i-- > 0; ) {

		// If the node is not an input (has no dependencies)
		if(nodes[i]->dependencies.size() != 0) {
			nodes.erase(nodes.begin() + i);
		}

		// Input node
		else {
			std::shared_ptr<ts::InputNode<T>> inputPtr =
			std::static_pointer_cast<ts::InputNode<T>>(nodes[i]);

			// If the node is not optimizable (has a null optimizedTensor)
			if(!(inputPtr->optimizedTensor)) {
				nodes.erase(nodes.begin() + i);
			}
		}
	}

	// Second pass : update tensors indices
	for(unsigned i = nodes.size(); i-- > 0; ) {
		std::shared_ptr<ts::InputNode<T>> inputPtr =
		std::static_pointer_cast<ts::InputNode<T>>(nodes[i]);

		inputPtr->optimizedTensor->index = i;
	}


	return nodes.size();
}



template <typename T>
void ts::WengertList<T>::toggleOptimize(ts::Tensor<T> * tensor, bool enable) {

	std::shared_ptr<ts::InputNode<T>> inputPtr =
	std::static_pointer_cast<ts::InputNode<T>>(nodes[tensor->index]);

	if(enable) {
		inputPtr->optimizedTensor = tensor;
	}
	else {
		inputPtr->optimizedTensor = NULL;
	}
}



	// ts::Tensor

// Input and not optimizable
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

		// Node without dependencies (input var,)
		std::shared_ptr<ts::Node<T>> nodePtr (
			new ts::InputNode<T>({newValue.rows(), newValue.cols()})
		);

		wList->nodes.push_back(nodePtr);
	} else {
		index = -1;
	}
};



// Tensor with dependencies, not optimizable
template <typename T>
ts::Tensor<T>::Tensor(
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> newValue,
	ts::WengertList<T> * newWList, std::shared_ptr<ts::Node<T>> node
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
// (not optimizable)
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
		derivatives[i].setZero(wList->nodes[i]->rows, wList->nodes[i]->cols);
	}

	// Initialize gradient of self with respect to itself
	derivatives[index].fill(1.0);


	// Iterate over the Wengert list backwards
	for (unsigned i = wList->nodes.size(); i-- > 0; ) {

		std::shared_ptr<ts::Node<T>> node = wList->nodes[i];

		// Increment parent nodes
		for(unsigned j = 0; j < node->dependencies.size(); j++) {
			derivatives[node->dependencies[j]] += node->incrementGradient(
				derivatives[i], j
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
