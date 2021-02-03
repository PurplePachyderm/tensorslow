/*
* General automatic differentiation engine based on a Wengert list
* implementation. Reverse mode only.
*/

#pragma once

#include <vector>
#include <memory>
#include <iostream>

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
	ts::Tensor<T> relu(const ts::Tensor<T> &x);
	template <typename T>
	ts::Tensor<T> leakyRelu(const ts::Tensor<T> &x);
	template <typename T>
	ts::Tensor<T> rescale(const ts::Tensor<T> &x);
	template <typename T>
	ts::Tensor<T> squaredNorm(const ts::Tensor<T> &x);


	// Forward declaration of friends
	// (grad accumulators and other autodiff operations)
	template <typename T> class GaElement;
	template <typename T> class GradientAccumulator;
	template <typename T> class AdamOptimizer;

	enum class ChannelSplit : int;

	template <typename T>
	ts::Tensor<T> convolution(const ts::Tensor<T> &mat, const ts::Tensor<T> &ker);

	template <typename T>
	ts::Tensor<T> maxPooling(const ts::Tensor<T> &x, std::vector<unsigned> pool);

	template <typename T>
	std::vector<ts::Tensor<T>> split(
		const ts::Tensor<T> &x,
		ChannelSplit channelSplit,
		unsigned nInputChannels
	);

	template <typename T>
	ts::Tensor<T> vertCat(const std::vector<ts::Tensor<T>> &x);

	template <typename T>
	ts::Tensor<T> flattening(const ts::Tensor<T> &x);

	template <typename T>
	ts::Tensor<T> im2col(
		const std::vector<ts::Tensor<T>> &x,
		std::vector<unsigned> kernelDim
	);

	template <typename T>
	std::vector<ts::Tensor<T>> col2im(
		const ts::Tensor<T> &x,
		std::vector<unsigned> outputDim
	);
}



	// ts::Node

template <typename T>
class ts::Node {
protected:

	Node() {}

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

	std::vector< Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> > values{};

	// Shape of the corresponding tensor
	long rows, cols;

public:

	friend ts::Tensor<T>;
	friend ts::WengertList<T>;
	friend ts::GradientAccumulator<T>;
	friend ts::AdamOptimizer<T>;	// Needed to initialize moment estimates

	friend ts::Tensor<T> operator+<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> operator-<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> operator*<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> operator/<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);

	friend ts::Tensor<T> matProd<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> sigmoid<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> relu<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> leakyRelu<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> rescale<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> squaredNorm<>(const ts::Tensor<T> &x);

	friend ts::Tensor<T> convolution<>(const ts::Tensor<T> &mat, const ts::Tensor<T> &ker);
	friend ts::Tensor<T> maxPooling<>(const ts::Tensor<T> &x, std::vector<unsigned> pool);
	friend std::vector<ts::Tensor<T>> split<>(
		const ts::Tensor<T> &x,
		ChannelSplit channelSplit,
		unsigned nInputChannels
	);
	friend ts::Tensor<T> vertCat<>(const std::vector<ts::Tensor<T>> &x);
	friend ts::Tensor<T> flattening<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> im2col<>(
		const std::vector<ts::Tensor<T>> &x,
		std::vector<unsigned> kernelDim
	);
	friend std::vector<ts::Tensor<T>> col2im<>(
		const ts::Tensor<T> &x,
		std::vector<unsigned> outputDim
	);

};



template <typename T>
class ts::InputNode : public ts::Node<T> {
private:
	using ts::Node<T>::Node;
	InputNode(std::vector<long> shape, bool model);

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> incrementGradient(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
			unsigned &j
	);

	// We will need this to optimize the tensor value in a ts::Model
	ts::Tensor<T> * optimizedTensor = NULL;

	// If true, node won't be removed on wList reset
	bool isModel = false;

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

	MatProdNode(
		std::vector<long> shape,
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> xVal, int xDep,
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> yVal, int yDep,
		std::vector<long int> newXSize, std::vector<long int> newYSize
	);

	// Size of the operands to figure out how to increment their partial
	// derivatives
	std::vector<long int> xSize;
	std::vector<long int> ySize;

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> incrementGradient(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
			unsigned &j
	);

	friend ts::Tensor<T> matProd<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
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
	bool elementWiseOnly = true;
	std::vector< std::shared_ptr<ts::Node<T>> > nodes{};

public:
	int size();
	int reset();

	// Make a tensor optimizable
	void toggleOptimize(ts::Tensor<T> * tensor, bool enable);

	friend class ts::Tensor<T>;
	friend class ts::GradientAccumulator<T>;
	friend class ts::AdamOptimizer<T>;	// Needed to initialize moment estimates

	// Other non-element wise operations (to change elementWiseOnly)
	friend ts::Tensor<T> matProd<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> sigmoid<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> relu<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> leakyRelu<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> rescale<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> squaredNorm<>(const ts::Tensor<T> &x);

	friend ts::Tensor<T> convolution<>(const ts::Tensor<T> &mat, const ts::Tensor<T> &ker);
	friend ts::Tensor<T> maxPooling<>(const ts::Tensor<T> &x, std::vector<unsigned> pool);
	friend std::vector<ts::Tensor<T>> split<>(
		const ts::Tensor<T> &x,
		ChannelSplit channelSplit,
		unsigned nInputChannels
	);
	friend ts::Tensor<T> vertCat<>(const std::vector<ts::Tensor<T>> &x);
	friend ts::Tensor<T> flattening<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> im2col<>(
		const std::vector<ts::Tensor<T>> &x,
		std::vector<unsigned> kernelDim
	);
	friend std::vector<ts::Tensor<T>> col2im<>(
		const ts::Tensor<T> &x,
		std::vector<unsigned> outputDim
	);
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

	Tensor() {};

	// Input tensor, part of model
	Tensor(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> newValue,
		ts::WengertList<T> * newWList
	);

	// Non part of model input tensor
	// (equivalent to calling previous constructor with model = false)
	Tensor(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> newValue,
		ts::WengertList<T> * newWList, bool model
	);

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> getValue();
	ts::Gradient<T> grad();


	friend ts::WengertList<T>;

	friend ts::Gradient<T>;
	friend ts::GaElement<T>;
	friend ts::GradientAccumulator<T>;

	friend ts::Tensor<T> operator+<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> operator-<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> operator*<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> operator/<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);

	friend ts::Tensor<T> matProd<>(const ts::Tensor<T> &x, const ts::Tensor<T> &y);
	friend ts::Tensor<T> sigmoid<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> relu<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> leakyRelu<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> rescale<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> squaredNorm<>(const ts::Tensor<T> &x);

	friend ts::Tensor<T> convolution<>(const ts::Tensor<T> &mat, const ts::Tensor<T> &ker);
	friend ts::Tensor<T> maxPooling<>(const ts::Tensor<T> &x, std::vector<unsigned> pool);
	friend std::vector<ts::Tensor<T>> split<>(
		const ts::Tensor<T> &x,
		ChannelSplit channelSplit,
		unsigned nInputChannels
	);
	friend ts::Tensor<T> vertCat<>(const std::vector<ts::Tensor<T>> &x);
	friend ts::Tensor<T> flattening<>(const ts::Tensor<T> &x);
	friend ts::Tensor<T> im2col<>(
		const std::vector<ts::Tensor<T>> &x,
		std::vector<unsigned> kernelDim
	);
	friend std::vector<ts::Tensor<T>> col2im<>(
		const ts::Tensor<T> &x,
		std::vector<unsigned> outputDim
	);
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
	friend class ts::AdamOptimizer<T>;
};
