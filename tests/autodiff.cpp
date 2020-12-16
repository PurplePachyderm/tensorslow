/*
* Test suite for the audodiff engine.
* We start with tests on scalar tensors, and go on with more and more complex
* examples. The last test is a minimal feedforward NN and is similar to
* our real use case for this small framework.
*/

#include <gtest/gtest.h>
#include <iostream>
#include <math.h>

#include "../include/tensorslow.h"



TEST(Operators, SimpleSum) {
	// Simple test of + operator (on scalars)
	ts::WengertList<float> wList;

	Eigen::ArrayXf a_(1,1);
	a_(0, 0) = (float) rand()/(float)(RAND_MAX/100.0);
	ts::Tensor<float> a = ts::Tensor<float>(a_, &wList);

	Eigen::ArrayXf b_(1,1);
	b_(0, 0) = (float) rand()/(float)(RAND_MAX/100.0);
	ts::Tensor<float> b = ts::Tensor<float>(b_, &wList);

	ts::Tensor<float> res = a + b;

	ts::Gradient<float> grad = res.grad();

	Eigen::Array<float, 1, 1> gradA = grad.getValue(a);
	Eigen::Array<float, 1, 1> gradB = grad.getValue(b);

	EXPECT_EQ(gradA(0, 0), 1.0);
	EXPECT_EQ(gradB(0, 0), 1.0);
	EXPECT_EQ(wList.size(), 3);
}



TEST(Operators, SimpleDiff) {
	// Simple test of - operator (on scalars)
	ts::WengertList<float> wList;

	Eigen::ArrayXf a_(1,1);
	a_(0, 0) = (float) rand()/(float)(RAND_MAX/100.0);
	ts::Tensor<float> a = ts::Tensor<float>(a_, &wList);

	Eigen::ArrayXf b_(1,1);
	b_(0, 0) = (float) rand()/(float)(RAND_MAX/100.0);
	ts::Tensor<float> b = ts::Tensor<float>(b_, &wList);

	ts::Tensor<float> res = a - b;

	ts::Gradient<float> grad = res.grad();

	Eigen::Array<float, 1, 1> gradA = grad.getValue(a);
	Eigen::Array<float, 1, 1> gradB = grad.getValue(b);

	EXPECT_EQ(gradA(0, 0), 1.0);
	EXPECT_EQ(gradB(0, 0), -1.0);
	EXPECT_EQ(wList.size(), 3);
}



TEST(Operators, SimpleProd) {
	// Simple test of * operator (on scalars)
	ts::WengertList<float> wList;

	Eigen::ArrayXf a_(1,1);
	a_(0,0) = (float) rand()/(float)(RAND_MAX/100.0);
	ts::Tensor<float> a = ts::Tensor<float>(a_, &wList);

	Eigen::ArrayXf b_(1,1);
	b_(0,0) = (float) rand()/(float)(RAND_MAX/100.0);
	ts::Tensor<float> b = ts::Tensor<float>(b_, &wList);

	ts::Tensor<float> res = a * b;

	ts::Gradient<float> grad = res.grad();

	Eigen::Array<float, 1, 1> gradA = grad.getValue(a);
	Eigen::Array<float, 1, 1> gradB = grad.getValue(b);

	EXPECT_EQ(gradA(0, 0), b.getValue()(0, 0));
	EXPECT_EQ(gradB(0, 0), a.getValue()(0, 0));
	EXPECT_EQ(wList.size(), 3);
}



TEST(Operators, SimpleDiv) {
	// Simple test of / operator (on scalars)
	ts::WengertList<float> wList;

	Eigen::ArrayXf a_(1,1);
	a_(0, 0) = (float) rand()/(float)(RAND_MAX/100.0);
	ts::Tensor<float> a = ts::Tensor<float>(a_, &wList);

	Eigen::ArrayXf b_(1,1);
	b_(0,0) = (float) rand()/(float)(RAND_MAX/100.0);
	ts::Tensor<float> b = ts::Tensor<float>(b_, &wList);

	ts::Tensor<float> res = a / b;

	ts::Gradient<float> grad = res.grad();

	EXPECT_EQ(grad.getValue(a)(0, 0), 1.0f / b.getValue()(0, 0));
	EXPECT_EQ(
		grad.getValue(b)(0, 0),
		- a.getValue()(0, 0) / (b.getValue()(0, 0) * b.getValue()(0, 0))
	);
	EXPECT_EQ(wList.size(), 3);
}



TEST(AutodiffTest, Polynomial) {
	// Slightly more complex example with a polynomial of degree 2 (on scalars)
	ts::WengertList<float> wList;

	Eigen::ArrayXf x_(1,1);
	x_(0, 0) = (float) rand()/(float)(RAND_MAX/100.0f);
	ts::Tensor<float> x = ts::Tensor<float>(x_, &wList);

	Eigen::ArrayXf a_(1,1);
	a_(0, 0) = (float) rand()/(float)(RAND_MAX/10.0f);
	ts::Tensor<float> a = ts::Tensor<float>(a_, &wList);

	Eigen::ArrayXf b_(1,1);
	b_(0, 0) = (float) rand()/(float)(RAND_MAX/10.0f);
	ts::Tensor<float> b = ts::Tensor<float>(b_, &wList);

	Eigen::ArrayXf c_(1,1);
	c_(0, 0) = (float) rand()/(float)(RAND_MAX/10.0f);
	ts::Tensor<float> c = ts::Tensor<float>(c_, &wList);

	ts::Tensor<float> y = a * x * x + b * x - c;

	ts::Gradient<float> grad = y.grad();


	EXPECT_EQ(
		y.getValue()(0, 0),
		a.getValue()(0, 0) * x.getValue()(0, 0) * x.getValue()(0, 0) +
		b.getValue()(0, 0) * x.getValue()(0, 0) - c.getValue()(0, 0)
	);
	EXPECT_EQ(
		grad.getValue(x)(0, 0),
		2.0f * a.getValue()(0, 0) * x.getValue()(0, 0) + b.getValue()(0, 0)
	);
	EXPECT_EQ(wList.size(), 9);
}



TEST(AutodiffTest, DifferentLists) {
	// Ensures that using  Tensors from different lists works
	// as expected

	ts::WengertList<float> wList1;
	ts::WengertList<float> wList2;

	Eigen::ArrayXf a_(1,1);
	a_(0,0) = (float) rand()/(float)(RAND_MAX/100.0);
	ts::Tensor<float> a = ts::Tensor<float>(a_, &wList1);

	Eigen::ArrayXf b_(1,1);
	b_(0,0) = (float) rand()/(float)(RAND_MAX/100.0);
	ts::Tensor<float> b = ts::Tensor<float>(b_, &wList2);

	ts::Tensor<float> c = a + b;

	EXPECT_EQ(c.getValue().rows(), 0.0);
	EXPECT_EQ(c.getValue().cols(), 0.0);
	EXPECT_EQ(wList1.size(), 1);
	EXPECT_EQ(wList2.size(), 1);
}



TEST(AutodiffTest, ElementWise) {
	// Tests a few element-wise operations on non scalar inputs

	ts::WengertList<float> wList;


	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> a_;
	a_.setRandom(3, 3);
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> b_;
	b_.setRandom(3, 3);
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> c_;
	c_.setRandom(3, 3);


	ts::Tensor<float> a = ts::Tensor<float>(a_, &wList);
	ts::Tensor<float> b = ts::Tensor<float>(b_, &wList);
	ts::Tensor<float> c = ts::Tensor<float>(c_, &wList);

	ts::Tensor<float> d = a * b + c;


	ts::Gradient<float> grad = d.grad();


	for(unsigned i=0; i<3; i++) {
		for(unsigned j=0; j<3; j++) {
			EXPECT_EQ(
				d.getValue()(i, j),
				a.getValue()(i,j) * b.getValue()(i,j) + c.getValue()(i,j)
			);

			EXPECT_EQ(
				grad.getValue(a)(i, j),
				b.getValue()(i,j)
			);

			EXPECT_EQ(
				grad.getValue(c)(i, j),
				1.0f
			);
		}
	}
}



TEST(AutodiffTest, Relu) {
	// Test of ReLU function

	ts::WengertList<float> wList;

	Eigen::Array<float, 3, 3> x_;
	x_ <<
	-1, 41, -42,
	12,  0,   0,
	 0, 42,   7;
	ts::Tensor<float> x = ts::Tensor<float>(x_, &wList);

	ts::Tensor<float> res = ts::relu(x);


	// Expected result and derivative
	Eigen::Array<float, 3, 3> expectedRes;
	expectedRes <<
	0, 0.976, 0,
	0.286, 0, 0,
	0, 1, 0.167;

	Eigen::Array<float, 3, 3> expectedDx;
	expectedDx <<
	0, 1.0f/42.0f, 0,
	1.0f/42.0f, 0, 0,
	0, 1.0f/42.0f, 1.0f/42.0f;


	// Get derivative
	ts::Gradient<float> grad = res.grad();
	Eigen::Array<float, 3, 3> dx = grad.getValue(x);


	// Compare results
	for(unsigned i=0; i<x_.rows(); i++) {
		for(unsigned j=0; j<x_.cols(); j++) {
			ASSERT_NEAR(res.getValue()(i, j), expectedRes(i,j), 0.001);
			ASSERT_NEAR(dx(i, j), expectedDx(i,j), 0.001);
		}
	}
}



TEST(AutodiffTest, MatProd) {
	// Tests a matrix-matrix product, as well as the gradient protection
	// mechanism (when computing grad of a non scalar tensor)

	ts::WengertList<float> wList;


	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> a_;
	a_.setRandom(3, 3);
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> b_;
	b_.setRandom(3, 3);
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> c_;
	c_ = a_.matrix() * b_.matrix();


	ts::Tensor<float> a = ts::Tensor<float>(a_, &wList);
	ts::Tensor<float> b = ts::Tensor<float>(b_, &wList);

	ts::Tensor<float> c = ts::matProd(a, b);


	ts::Gradient<float> grad = c.grad();


	for(unsigned i=0; i<3; i++) {
		for(unsigned j=0; j<3; j++) {
			EXPECT_EQ(c_(i, j), c.getValue()(i,j));
			EXPECT_EQ(grad.isEmpty(), true);
		}
	}
}



TEST(AutodiffTest, SimpleNN) {
	// Simulates a simple feedforward neural network with no hidden layer, and
	// its cost function. We'll compute the gradient of this function on a
	// graph having element element-wise operations, a matrix product and a norm.

	ts::WengertList<float> wList;


	// Initialize all input elements

	Eigen::Array<float, 2, 1> inputLayer_;
	inputLayer_ <<
	0.6,
	0.4;
	ts::Tensor<float> inputLayer = ts::Tensor<float>(inputLayer_, &wList);

	Eigen::Array<float, 3, 2> weights_;
	weights_ <<
	0.5, 0.5,
	2.0, 3.0,
	0.0, 6.0;
	ts::Tensor<float> weights = ts::Tensor<float>(weights_, &wList);

	Eigen::Array<float, 3, 1> biases_;
	biases_ <<
	-0.2,
	0.2,
	0.3;
	ts::Tensor<float> biases = ts::Tensor<float>(biases_, &wList);

	Eigen::Array<float, 3, 1> target_;
	target_ <<
	0.0,
	1.0,
	0.0;
	ts::Tensor<float> target = ts::Tensor<float>(target_, &wList);


	// Initialize expected values (pre computed using PyTorch's Autograd)

	Eigen::Array<float, 3,1> expectedOutputLayer;
	expectedOutputLayer <<
	0.5744,
	0.9309,
	0.9370;

	float expectedCost = 1.2128;

	Eigen::Array<float, 3, 2> expectedWeightsGrad;
	expectedWeightsGrad <<
	0.1685, 0.1123,
	-0.0053, -0.0036,
	0.0664, 0.0442;

	Eigen::Array<float, 3, 1> expectedBiasesGrad;
	expectedBiasesGrad <<
	0.2809,
	-0.0089,
	0.1106;


	// Compute cost

	ts::Tensor<float> outputLayer = ts::matProd(weights, inputLayer);
	outputLayer = ts::sigmoid(outputLayer + biases);

	ts::Tensor<float> cost = ts::squaredNorm(outputLayer - target);


	// Compute grad of cost function

	ts::Gradient<float> grad = cost.grad();

	Eigen::Array<float, 3, 2> weigthsGrad = grad.getValue(weights);
	Eigen::Array<float, 3, 1> biasesGrad = grad.getValue(biases);


	// Compare with pre-computed result
	// Using NEAR because hardcoded values are approximate

	EXPECT_NEAR(expectedCost, cost.getValue()(0,0), 0.0001);

	for(unsigned i=0; i<3; i++) {
		for(unsigned j=0; j<2; j++) {
			EXPECT_NEAR(weigthsGrad(i, j), expectedWeightsGrad(i, j), 0.0001);
		}
	}

	for(unsigned i=0; i<3; i++) {
		for(unsigned j=0; j<1; j++) {
			EXPECT_NEAR(biasesGrad(i, j), expectedBiasesGrad(i, j), 0.0001);
		}
	}
}



int main(int argc, char **argv) {
	std::cout << "*** AUTODIFF TEST SUITE ***" << std::endl;

	srand(time(NULL));

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
