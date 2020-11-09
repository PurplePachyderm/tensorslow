/*
* Test suite for the ts::Model class and its instantiations.
* The creation of a simple model will be tested, then more complex examples on
* neural networks models will follow.
*/

#include <gtest/gtest.h>
#include <iostream>
#include <math.h>

#include "../include/tensorslow.h"



TEST(ModelInstantiation, Polynom) {
	// TODO
}



TEST(MultiLayerPerceptron, ForwardPass) {
	// TODO
	ts::MultiLayerPerceptron<float> model(2, {3});
}



int main(int argc, char **argv) {
	std::cout << "*** MODELS TEST SUITE ***" << std::endl;

	srand(time(NULL));

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
