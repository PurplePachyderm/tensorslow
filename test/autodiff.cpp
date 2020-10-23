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
	ts::WengertList wList;

	ts::Var a = ts::Var((float)rand()/(float)(RAND_MAX/100.0), &wList);
	ts::Var b = ts::Var((float)rand()/(float)(RAND_MAX/100.0), &wList);

	ts::Var res = a + b;

	ts::Gradient grad = res.grad();

	float gradA = grad.getValue(a);
	float gradB = grad.getValue(b);

	ASSERT_EQ(gradA, 1.0);
	ASSERT_EQ(gradB, 1.0);
	ASSERT_EQ(wList.size(), 3);
}



TEST(AutodiffTest, SimpleDiff) {
	// Simple test of - operator
	ts::WengertList wList;

	ts::Var a = ts::Var((float)rand()/(float)(RAND_MAX/100.0), &wList);
	ts::Var b = ts::Var((float)rand()/(float)(RAND_MAX/100.0), &wList);

	ts::Var res = a - b;

	ts::Gradient grad = res.grad();

	float gradA = grad.getValue(a);
	float gradB = grad.getValue(b);

	ASSERT_EQ(gradA, 1.0);
	ASSERT_EQ(gradB, -1.0);
	ASSERT_EQ(wList.size(), 3);
}



TEST(AutodiffTest, SimpleProd) {
	// Simple test of * operator
	ts::WengertList wList;

	ts::Var a = ts::Var((float)rand()/(float)(RAND_MAX/100.0), &wList);
	ts::Var b = ts::Var((float)rand()/(float)(RAND_MAX/100.0), &wList);

	ts::Var res = a * b;

	ts::Gradient grad = res.grad();

	float gradA = grad.getValue(a);
	float gradB = grad.getValue(b);

	ASSERT_EQ(gradA, b.getValue());
	ASSERT_EQ(gradB, a.getValue());
	ASSERT_EQ(wList.size(), 3);
}



TEST(AutodiffTest, SimpleDiv) {
	// Simple test of / operator
	ts::WengertList wList;

	ts::Var a = ts::Var((float)rand()/(float)(RAND_MAX/100.0), &wList);
	ts::Var b = ts::Var((float)rand()/(float)(RAND_MAX/100.0), &wList);

	ts::Var res = a / b;

	ts::Gradient grad = res.grad();

	ASSERT_EQ(grad.getValue(a), 1.0f / b.getValue());
	ASSERT_EQ(grad.getValue(b), - a.getValue() / (b.getValue() * b.getValue()));
	ASSERT_EQ(wList.size(), 3);
}



TEST(AutodiffTest, Polynomial) {
	// Slightly more complex example with a polynomial of degree 2

	std::cout << std::setprecision(17);

	ts::WengertList wList;

	ts::Var x = ts::Var((float)rand()/(float)(RAND_MAX/100.0f), &wList);

	ts::Var a = ts::Var((float)rand()/(float)(RAND_MAX/10.0f), &wList);
	ts::Var b = ts::Var((float)rand()/(float)(RAND_MAX/10.0f), &wList);
	ts::Var c = ts::Var((float)rand()/(float)(RAND_MAX/10.0f), &wList);

	ts::Var y = a * x * x + b * x - c;

	ts::Gradient grad = y.grad();


	ASSERT_EQ(y.getValue(), a.getValue() * x.getValue() * x.getValue() + b.getValue() * x.getValue() - c.getValue());
	ASSERT_EQ(grad.getValue(x), 2.0f * a.getValue() * x.getValue() + b.getValue());
	ASSERT_EQ(wList.size(), 9);
}



TEST(AutodiffTest, DifferentLists) {
	// Ensures that using  Vars from different lists works
	// as expected

	ts::WengertList wList1;
	ts::WengertList wList2;

	ts::Var a = ts::Var((float)rand()/(float)(RAND_MAX/100.0), &wList1);
	ts::Var b = ts::Var((float)rand()/(float)(RAND_MAX/100.0), &wList2);

	ts::Var c = a + b;

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
