/*
* Test suite for the serialization utility functions, as well as prebuilt
* models save and load functions.
*/

#include <gtest/gtest.h>
#include <iostream>

#include "../include/tensorslow.h"



TEST(Utilities, Tensor) {

	// Try to save a tensor to a file and load it to a new tensor

	ts::WengertList<float> wList;

	// Create base tensor
	Eigen::Array<float, 3, 4> a_;
	a_ <<
	0, 1.1, 2.2, 3.3,
	4.4, 5.5, 6.6, 7.7,
	8.8, 9.9, 10.10, 11.11;


	ts::Tensor<float> a(a_, &wList);

	// Write tensor to file
	std::ofstream out("tests/tensor.ts");
	out << ts::serializeTensor(a);
	out.close();

	std::cout << "Yeah" << std::endl;

	// Read and copy tensor to other file
	std::ifstream in("tests/tensor.ts");

	ts::Tensor<float> b = ts::parseTensor(in, &wList);


	// Compare values
	ASSERT_EQ(a.getValue().rows(), b.getValue().rows());
	ASSERT_EQ(a.getValue().cols(), b.getValue().cols());

	for(unsigned i=0; i<3; i++) {
		for(unsigned j=0; j<4; j++) {
			EXPECT_EQ(a.getValue()(i, j), b.getValue()(i, j));
		}
	}
}



TEST(Utilities, Vector) {
	// Try to save a tensors vector to a file and load it to a new vector
}



int main(int argc, char **argv) {
	std::cout << "*** SERIALIZER TEST SUITE ***" << std::endl;

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
