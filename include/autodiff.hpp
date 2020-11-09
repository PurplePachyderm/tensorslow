/*
* General automatic differentiation engine based on a Wengert list
* implementation. Reverse mode only.
*/

#pragma once

#include <vector>
#include <memory>

#include <Eigen/Dense>



namespace ts {
	template <typename T> class Node;
	template <typename T> class InputNode;
	template <typename T> class ElementWiseNode;
	template <typename T> class MatProdNode;
	template <typename T> class ScalarNode;

	template <typename T> class WengertList;
	template <typename T> class Tensor;
	template <typename T> class Gradient;


	// This helper function allows us to create Tensor instances without
	// template syntax. This way, the type will be the same as its parent
	// WengertList.

	template <typename T>
	ts::Tensor<T> NewTensor(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> newValue,
		ts::WengertList<T> * newWList
	);

	template <typename T>
	ts::Tensor<T> operator+(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	template <typename T>
	ts::Tensor<T> operator-(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	template <typename T>
	ts::Tensor<T> operator*(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	template <typename T>
	ts::Tensor<T> operator/(const ts::Tensor<T> &x, const ts::Tensor<T> &y);

	template <typename T>
	ts::Tensor<T> matProd(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	template <typename T>
	ts::Tensor<T> sigmoid(const ts::Tensor<T> &x);
	template <typename T>
	ts::Tensor<T> squaredNorm(const ts::Tensor<T> &x);


	// Forward declaration of friends (not related to audodiff)
	template <typename T> class GaElement;
	template <typename T> class GradientAccumulator;
}



	// ts::Node

template <typename T>
class ts::Node {
private:
	// Represents an input variable
	Node(std::vector<long> shape);

	// Represents a unary operator
	Node(std::vector<long> shape,
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> xVal, int xDep
	);

	// Represents a binary operator
	Node(
		std::vector<long> shape,
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> xVal, int xDep,
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> yVal, int yDep
	);


	std::vector<int> dependencies{};


	virtual Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> incrementGradient(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
			unsigned &j
	) = 0;

protected:
	std::vector< Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> > values{};
	// Shape of the corresponding tensor
	long rows, cols;

public:

	friend ts::Tensor<T>;
	friend ts::WengertList<T>;
	friend ts::GradientAccumulator<T>;

	friend ts::Tensor<T> operator+<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> operator-<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> operator*<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> operator/<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);

	friend ts::Tensor<T> matProd<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> sigmoid<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> squaredNorm<>(const ts::Tensor<T> &x);

};



template <typename T>
class ts::InputNode : public ts::Node<T> {
private:
	using ts::Node<T>::Node;

	// We will need this to optimize the tensor value in a ts::Model
	ts::Tensor<T> * optimizedTensor = NULL;

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> incrementGradient(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
			unsigned &j
	);

public:

	friend ts::WengertList<T>;
	friend ts::Tensor<T>;
	friend ts::GradientAccumulator<T>;
};



template <typename T>
class ts::ElementWiseNode : public ts::Node<T> {
private:
	using ts::Node<T>::Node;

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> incrementGradient(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
			unsigned &j
	);
};



template <typename T>
class ts::MatProdNode : public ts::Node<T> {
private:
	using ts::Node<T>::Node;

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> incrementGradient(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
			unsigned &j
	);
};



template <typename T>
class ts::ScalarNode : public ts::Node<T> {
private:
	using ts::Node<T>::Node;

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> incrementGradient(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
			unsigned &j
	);
};



	// ts::WengertList

template <typename T>
class ts::WengertList {
private:
	std::vector< std::shared_ptr<ts::Node<T>> > nodes{};
	bool elementWiseOnly = true;

public:
	int size();
	int reset();

	friend class ts::Tensor<T>;
	friend class ts::GradientAccumulator<T>;

	// Other non-element wise operations (to change elementWiseOnly)
	friend ts::Tensor<T> matProd<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> sigmoid<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> squaredNorm<>(const ts::Tensor<T> &x);
};



	// ts::Tensor

template <typename T>
class ts::Tensor {
private:
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> value;
	ts::WengertList<T> * wList = NULL;
	int index;

	// We want this constructor to be private as it is supposed to be called by
	// our friends overloaded operators and functions only. This constructor
	// thus allows us to create a Tensor with dependencies in the Wengert list.
	Tensor(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> newValue,
		ts::WengertList<T> * newWList, std::shared_ptr<ts::Node<T>> node
	);

public:

	// Input tensor might be optimizable (if boolean is true) -> only for tensors
	// that are part of a model parameter
	// WARNING Should be called in ts::Member derived classes only
	// (made public for now => create interface class later ?)
	Tensor(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> newValue,
		ts::WengertList<T> * newWList, bool optimizable
	);

	// Non optimizable input tensor (calling previous constructor with
	// optimizable = false)
	Tensor(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> newValue,
		ts::WengertList<T> * newWList
	);

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> getValue();
	ts::Gradient<T> grad();


	friend ts::Gradient<T>;
	friend ts::GaElement<T>;
	friend ts::GradientAccumulator<T>;

	friend ts::Tensor<T> operator+<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> operator-<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> operator*<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> operator/<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);

	friend ts::Tensor<T> matProd<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> sigmoid<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> squaredNorm<>(const ts::Tensor<T> &x);
};



template <typename T>
ts::Tensor<T> NewTensor(
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> newValue,
	ts::WengertList<T> * newWList
);



	// ts::Gradient

template <typename T>
class ts::Gradient {
private:
	// Constructor is private since we want instances of this class to be
	// generated by the Tensor::grad() method only
	Gradient(
		std::vector< Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> > newDerivatives
	);

	std::vector< Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> > derivatives;

public:
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> getValue(ts::Tensor<T> a);
	bool isEmpty();

	friend class ts::Tensor<T>;
	friend class ts::GradientAccumulator<T>;
};
