/*
* Test suite for different optimizers.
* This test suite will contain tests on simple models using different
* optimizers to ensure the convergence of the loss function.
*/

#include <gtest/gtest.h>
#include <iostream>
#include <math.h>

#include "../include/tensorslow.h"



TEST(GradientDescent, Polynom) {
	// TODO
	ts::GradientDescentOptimizer<float> optimizer();
}



int main(int argc, char **argv) {
	std::cout << "*** OPTIMIZERS TEST SUITE ***" << std::endl;

	srand(time(NULL));

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
