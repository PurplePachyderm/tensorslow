/*
* Test suite for the audodiff engine
*/

#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
#include <math.h>

#include "../include/tensorslow.h"



TEST(AutodiffTest, SimpleSum) {
	// Simple test of + operator

	ts::WengertList<float> wList;

	auto a = ts::NewVar((float)rand()/(float)(RAND_MAX/100.0), &wList);
	auto b = ts::NewVar((float)rand()/(float)(RAND_MAX/100.0), &wList);

	auto res = a + b;

	auto grad = res.grad();

	auto gradA = grad.getValue(a);
	auto gradB = grad.getValue(b);

	ASSERT_EQ(gradA, 1.0);
	ASSERT_EQ(gradB, 1.0);
	ASSERT_EQ(wList.size(), 3);
}



TEST(AutodiffTest, SimpleDiff) {
	// Simple test of - operator

	ts::WengertList<float> wList;

	auto a = ts::NewVar((float)rand()/(float)(RAND_MAX/100.0), &wList);
	auto b = ts::NewVar((float)rand()/(float)(RAND_MAX/100.0), &wList);

	auto res = a - b;

	auto grad = res.grad();

	auto gradA = grad.getValue(a);
	auto gradB = grad.getValue(b);

	ASSERT_EQ(gradA, 1.0);
	ASSERT_EQ(gradB, -1.0);
	ASSERT_EQ(wList.size(), 3);
}



TEST(AutodiffTest, SimpleProd) {
	// Simple test of * operator

	ts::WengertList<float> wList;

	auto a = ts::NewVar((float)rand()/(float)(RAND_MAX/100.0), &wList);
	auto b = ts::NewVar((float)rand()/(float)(RAND_MAX/100.0), &wList);

	auto res = a * b;

	auto grad = res.grad();

	auto gradA = grad.getValue(a);
	auto gradB = grad.getValue(b);

	ASSERT_EQ(gradA, b.getValue());
	ASSERT_EQ(gradB, a.getValue());
	ASSERT_EQ(wList.size(), 3);
}



TEST(AutodiffTest, SimpleDiv) {
	// Simple test of / operator

	ts::WengertList<float> wList;

	auto a = ts::NewVar((float)rand()/(float)(RAND_MAX/100.0), &wList);
	auto b = ts::NewVar((float)rand()/(float)(RAND_MAX/100.0), &wList);

	auto res = a / b;

	auto grad = res.grad();

	ASSERT_EQ(grad.getValue(a), 1.0f / b.getValue());
	ASSERT_EQ(grad.getValue(b), - a.getValue() / (b.getValue() * b.getValue()));
	ASSERT_EQ(wList.size(), 3);
}



TEST(AutodiffTest, Polynomial) {
	// Slightly more complex example with a polynomial of degree 2

	ts::WengertList<float> wList;

	auto x = ts::NewVar((float)rand()/(float)(RAND_MAX/100.0f), &wList);

	auto a = ts::NewVar((float)rand()/(float)(RAND_MAX/10.0f), &wList);
	auto b = ts::NewVar((float)rand()/(float)(RAND_MAX/10.0f), &wList);
	auto c = ts::NewVar((float)rand()/(float)(RAND_MAX/10.0f), &wList);

	auto y = a * x * x + b * x - c;

	auto grad = y.grad();


	ASSERT_EQ(y.getValue(), a.getValue() * x.getValue() * x.getValue() + b.getValue() * x.getValue() - c.getValue());
	ASSERT_EQ(grad.getValue(x), 2.0f * a.getValue() * x.getValue() + b.getValue());
	ASSERT_EQ(wList.size(), 9);
}



TEST(AutodiffTest, DifferentLists) {
	// Ensures that using  Vars from different lists works
	// as expected

	ts::WengertList<float> wList1;
	ts::WengertList<float> wList2;

	auto a = ts::NewVar((float)rand()/(float)(RAND_MAX/100.0), &wList1);
	auto b = ts::NewVar((float)rand()/(float)(RAND_MAX/100.0), &wList2);

	auto c = a + b;

	ASSERT_EQ(c.getValue(), 0.0);
	ASSERT_EQ(wList1.size(), 1);
	ASSERT_EQ(wList2.size(), 1);
}



int main(int argc, char **argv) {
	std::cout << "*** AUTODIFF TEST SUITE ***" << std::endl;

	srand (time(NULL));

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
