/*
* General automatic differentiation engine based on a Wengert list
* implementation. Reverse mode only.
*/

#include "../include/autodiff.hpp"


	// ts::Node

ts::Node::Node() {

};



ts::Node::Node(float xVal, int xDep) {
	values =  {xVal};	// [da/dx]
	dependencies =  {xDep};
}



ts::Node::Node(float xVal, int xDep, float yVal, int yDep) {
	values =  {xVal, yVal};	// [da/dx, da/dy]
	dependencies =  {xDep, yDep};
}



	// ts::WengertList

int ts::WengertList::size() {
	return nodes.size();
}



	// ts::Var

ts::Var::Var(float newValue, ts::WengertList * newWList) {
	value = newValue;

	wList = newWList;

	if(wList != NULL) {
		// Add new Var to the Wengert list
		index = wList->nodes.size();
		wList->nodes.push_back(ts::Node());	// Node without dependencies (input var)
	} else {
		index = -1;
	}
}



ts::Var::Var(float newValue, ts::WengertList * newWList, ts::Node node) {
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



float ts::Var::getValue() {
	return value;
}



ts::Gradient ts::Var::grad() {
	// Computes the gradient of all Wengert list's nodes with respect to this
	// variable. Derivatives are stored in a vector wich size equals the
	// Wengert list's.

	std::vector<float> derivatives(wList->nodes.size(), 0.0);
	derivatives[index] = 1.0;


	// Iterate over the Wengert list backwards
	for (unsigned i = wList->nodes.size(); i > 0; i--) {
		Node * node = &(wList->nodes[i]);
		float derivative = derivatives[i];

		// Increment parent nodes
		for(unsigned j = 0; j < node->dependencies.size(); j++) {
			derivatives[node->dependencies[j]] += node->values[j] * derivative;
		}
	}

	return ts::Gradient(derivatives);
}



	// ts::Gradient

ts::Gradient::Gradient(std::vector<float> newDerivatives) {
	derivatives = newDerivatives;
}

float ts::Gradient::getValue(ts::Var a) {
	return derivatives[a.index];
}



	// Overloaded arithmetic operators

ts::Var ts::operator+(const ts::Var &x, const ts::Var &y){
	if(x.wList != y.wList) {
		return ts::Var(0.0, NULL);
	}

	// a = x + y
	// da / dx = 1
	// a / dy = 1
	return ts::Var(
		x.value + y.value,
		x.wList,
		ts::Node(1.0, x.index, 1.0, y.index)
	);
}



ts::Var ts::operator-(const ts::Var &x, const ts::Var &y){
	if(x.wList != y.wList) {
		return ts::Var(0.0, NULL);
	}

	// a = x - y
	// da / dx = 1
	// a / dy = -1
	return ts::Var(
		x.value - y.value,
		x.wList,
		ts::Node(1.0, x.index, -1.0, y.index)
	);
}



ts::Var ts::operator*(const ts::Var &x, const ts::Var &y){
	if(x.wList != y.wList) {
		return Var(0.0, NULL);
	}

	// a = x * y
	// da / dx = y
	// a / dy = x
	return ts::Var(
		x.value * y.value,
		x.wList,
		ts::Node(y.value, x.index, x.value, y.index)
	);
}



ts::Var ts::operator/(const ts::Var &x, const ts::Var &y){
	if(x.wList != y.wList) {
		return ts::Var(0.0, NULL);
	}

	// a = x / y
	// da / dx = 1 / y
	// a / dy = -x / y^2
	return ts::Var(
		x.value + y.value,
		x.wList,
		ts::Node(1.0 / y.value, x.index, -x.value / (y.value * y.value) , y.index)
	);
}
