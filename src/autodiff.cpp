/*
* General automatic differentiation engine based on a Wengert list
* implementation. Reverse mode only.
*/

#include "../include/autodiff.hpp"


	// ts::Node

template <typename T>
ts::Node<T>::Node() {

};



template <typename T>
ts::Node<T>::Node(T xVal, int xDep) {
	values =  {xVal};	// [da/dx]
	dependencies =  {xDep};
}



template <typename T>
ts::Node<T>::Node(T xVal, int xDep, T yVal, int yDep) {
	values =  {xVal, yVal};	// [da/dx, da/dy]
	dependencies =  {xDep, yDep};
}



	// ts::WengertList

template <typename T>
int ts::WengertList<T>::size() {
	return nodes.size();
}



	// ts::Var

template <typename T>
ts::Var<T>::Var(T newValue, ts::WengertList<T> * newWList) {
	value = newValue;

	wList = newWList;

	if(wList != NULL) {
		// Add new Var to the Wengert list
		index = wList->nodes.size();
		wList->nodes.push_back(ts::Node<T>());	// Node without dependencies (input var)
	} else {
		index = -1;
	}
}



template <typename T>
ts::Var<T>::Var(T newValue, ts::WengertList<T> * newWList, ts::Node<T> node) {
	value = newValue;

	wList = newWList;

	if(wList != NULL) {
		// Add new Var to the Wengert list
		index = wList->nodes.size();
		wList->nodes.push_back(node);	// This node can contain dependencies & values
	} else {
		index = -1;
	}
}



// Helper function to create new instances without syntax template
template <typename T>
ts::Var<T> ts::NewVar(T newValue, ts::WengertList<T> * newWList) {
	return ts::Var<T>(newValue, newWList);
}



template <typename T>
T ts::Var<T>::getValue() {
	return value;
}



template <typename T>
ts::Gradient<T> ts::Var<T>::grad() {
	// Computes the gradient of all Wengert list's nodes with respect to this
	// variable. Derivatives are stored in a vector wich size equals the
	// Wengert list's.

	std::vector<T> derivatives(wList->nodes.size(), 0.0);
	derivatives[index] = 1.0;


	// Iterate over the Wengert list backwards
	for (unsigned i = wList->nodes.size(); i > 0; i--) {
		ts::Node<T> * node = &(wList->nodes[i]);
		T derivative = derivatives[i];

		// Increment parent nodes
		for(unsigned j = 0; j < node->dependencies.size(); j++) {
			derivatives[node->dependencies[j]] += node->values[j] * derivative;
		}
	}

	return ts::Gradient<T>(derivatives);
}



	// ts::Gradient

template <typename T>
ts::Gradient<T>::Gradient(std::vector<T> newDerivatives) {
	derivatives = newDerivatives;
}



template <typename T>
T ts::Gradient<T>::getValue(ts::Var<T> a) {
	return derivatives[a.index];
}



	// Overloaded arithmetic operators

template <typename T>
ts::Var<T> ts::operator+(const ts::Var<T> &x, const ts::Var<T> &y){
	if(x.wList != y.wList) {
		return ts::Var<T>(0.0, NULL);
	}

	// a = x + y
	// da / dx = 1
	// a / dy = 1
	return ts::Var<T>(
		x.value + y.value,
		x.wList,
		ts::Node<T>(1.0, x.index, 1.0, y.index)
	);
}



template <typename T>
ts::Var<T> ts::operator-(const ts::Var<T> &x, const ts::Var<T> &y){
	if(x.wList != y.wList) {
		return ts::Var<T>(0.0, NULL);
	}

	// a = x - y
	// da / dx = 1
	// a / dy = -1
	return ts::Var<T>(
		x.value - y.value,
		x.wList,
		ts::Node<T>(1.0, x.index, -1.0, y.index)
	);
}



template <typename T>
ts::Var<T> ts::operator*(const ts::Var<T> &x, const ts::Var<T> &y){
	if(x.wList != y.wList) {
		return Var<T>(0.0, NULL);
	}

	// a = x * y
	// da / dx = y
	// a / dy = x
	return ts::Var<T>(
		x.value * y.value,
		x.wList,
		ts::Node<T>(y.value, x.index, x.value, y.index)
	);
}



template <typename T>
ts::Var<T> ts::operator/(const ts::Var<T> &x, const ts::Var<T> &y){
	if(x.wList != y.wList) {
		return ts::Var<T>(0.0, NULL);
	}

	// a = x / y
	// da / dx = 1 / y
	// a / dy = -x / y^2
	return ts::Var<T>(
		x.value + y.value,
		x.wList,
		ts::Node<T>(1.0 / y.value, x.index, -x.value / (y.value * y.value), y.index)
	);
}



// Add data types support manually (the compiler would need an implementation
// directly in the header file otherwise).
// This method has the advantage of forbidding non-numeric data types like
// strings, etc... as well as keeping the source code in this .cpp file.
// More data types may be added in the future.

	// float
template class ts::Node<float>;
template class ts::WengertList<float>;
template class ts::Var<float>;
template class ts::Gradient<float>;
template ts::Var<float> ts::NewVar(float newValue, ts::WengertList<float> * newWList);
template ts::Var<float> ts::operator+(const ts::Var<float> &x, const ts::Var<float> &y);
template ts::Var<float> ts::operator-(const ts::Var<float> &x, const ts::Var<float> &y);
template ts::Var<float> ts::operator*(const ts::Var<float> &x, const ts::Var<float> &y);
template ts::Var<float> ts::operator/(const ts::Var<float> &x, const ts::Var<float> &y);

	// double
template class ts::Node<double>;
template class ts::WengertList<double>;
template class ts::Var<double>;
template class ts::Gradient<double>;
template ts::Var<double> ts::NewVar(double newValue, ts::WengertList<double> * newWList);
template ts::Var<double> ts::operator+(const ts::Var<double> &x, const ts::Var<double> &y);
template ts::Var<double> ts::operator-(const ts::Var<double> &x, const ts::Var<double> &y);
template ts::Var<double> ts::operator*(const ts::Var<double> &x, const ts::Var<double> &y);
template ts::Var<double> ts::operator/(const ts::Var<double> &x, const ts::Var<double> &y);
