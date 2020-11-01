/*
* Test suite for the audodiff engine
*/

#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
#include <math.h>

#include "../include/tensorslow.h"



TEST(AutodiffTest, SimpleSum) {
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

	ASSERT_EQ(gradA(0, 0), 1.0);
	ASSERT_EQ(gradB(0, 0), 1.0);
	ASSERT_EQ(wList.size(), 3);
}



TEST(AutodiffTest, SimpleDiff) {
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

	ASSERT_EQ(gradA(0, 0), 1.0);
	ASSERT_EQ(gradB(0, 0), -1.0);
	ASSERT_EQ(wList.size(), 3);
}



TEST(AutodiffTest, SimpleProd) {
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

	ASSERT_EQ(gradA(0, 0), b.getValue()(0, 0));
	ASSERT_EQ(gradB(0, 0), a.getValue()(0, 0));
	ASSERT_EQ(wList.size(), 3);
}



TEST(AutodiffTest, SimpleDiv) {
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

	ASSERT_EQ(grad.getValue(a)(0, 0), 1.0f / b.getValue()(0, 0));
	ASSERT_EQ(
		grad.getValue(b)(0, 0),
		- a.getValue()(0, 0) / (b.getValue()(0, 0) * b.getValue()(0, 0))
	);
	ASSERT_EQ(wList.size(), 3);
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


	ASSERT_EQ(
		y.getValue()(0, 0),
		a.getValue()(0, 0) * x.getValue()(0, 0) * x.getValue()(0, 0) +
		b.getValue()(0, 0) * x.getValue()(0, 0) - c.getValue()(0, 0)
	);
	ASSERT_EQ(
		grad.getValue(x)(0, 0),
		2.0f * a.getValue()(0, 0) * x.getValue()(0, 0) + b.getValue()(0, 0)
	);
	ASSERT_EQ(wList.size(), 9);
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

	ASSERT_EQ(c.getValue().rows(), 0.0);
	ASSERT_EQ(c.getValue().cols(), 0.0);
	ASSERT_EQ(wList1.size(), 1);
	ASSERT_EQ(wList2.size(), 1);
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
			ASSERT_EQ(
				d.getValue()(i, j),
				a.getValue()(i,j) * b.getValue()(i,j) + c.getValue()(i,j)
			);

			ASSERT_EQ(
				grad.getValue(a)(i, j),
				b.getValue()(i,j)
			);

			ASSERT_EQ(
				grad.getValue(c)(i, j),
				1.0f
			);
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
			ASSERT_EQ(c_(i, j), c.getValue()(i,j));
			ASSERT_EQ(grad.isEmpty(), true);
		}
	}
}



int main(int argc, char **argv) {
	std::cout << "*** AUTODIFF TEST SUITE ***" << std::endl;

	srand (time(NULL));

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
