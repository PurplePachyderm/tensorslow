/*
* Test suite for the audodiff engine
*/

#include <gtest/gtest.h>
#include <iostream>
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

	float gradA = grad.getValue(a);
	float gradB = grad.getValue(b);

	float valA = a.getValue();
	float valB = b.getValue();

	// Using ASSERT_NEAR because ASSERT_EQ returns false
	// (most definitely because of floating point errors)
	ASSERT_NEAR(gradA, 1.0 / valB, 0.0000001);
	ASSERT_NEAR(gradB, -valA / (valB * valB), 0.0000001);
	ASSERT_EQ(wList.size(), 3);
}



TEST(AutodiffTest, Polynomial) {
	// Slightly more complex example with a polynomial of degree 2

	ts::WengertList wList;

	ts::Var x = ts::Var((float)rand()/(float)(RAND_MAX/100.0), &wList);

	ts::Var a = ts::Var((float)rand()/(float)(RAND_MAX/10.0), &wList);
	ts::Var b = ts::Var((float)rand()/(float)(RAND_MAX/10.0), &wList);
	ts::Var c = ts::Var((float)rand()/(float)(RAND_MAX/10.0), &wList);

	ts::Var y = a * x * x + b * x - c;

	float valX = x.getValue();
	float valY = y.getValue();
	ts::Gradient grad = y.grad();
	float gradX = grad.getValue(x);

	// Using ASSERT_NEAR again
	// NOTE Rounding errors seem to get more important after every operation.
	// This could be a problem in the future, while evaluating complex
	// expressions. Test doesn't pass for now.
	ASSERT_NEAR(valY, a.getValue() * valX * valX + b.getValue() * valX - c.getValue(), 0.0000001);
	ASSERT_NEAR(gradX, 2.0 * a.getValue() * valX + b.getValue(), 0.0000001);
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
