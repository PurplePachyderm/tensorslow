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


	// Read and copy tensor to other file
	std::ifstream in("tests/tensor.ts");
	ts::Tensor<float> b = ts::parseTensor(in, &wList);
	in.close();


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

	ts::WengertList<float> wList;


	// Create base vector
	std::vector<ts::Tensor<float>> vecSrc = {};
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> arr;

	unsigned size = 10;
	unsigned rows = 5;
	unsigned cols = 5;

	for(unsigned i=0; i<size; i++) {
		arr.setRandom(rows, cols);
		vecSrc.push_back(ts::Tensor<float>(arr, &wList));
	}


	// Write vector to file
	std::ofstream out("tests/vector.ts");
	out << ts::serializeTensorsVector(vecSrc);
	out.close();


	// Read and copy vector to other file
	std::ifstream in("tests/vector.ts");
	std::vector<ts::Tensor<float>> vecDist = ts::parseTensorsVector(in, &wList);
	in.close();


	// Compare values
	ASSERT_EQ(vecSrc.size(), vecDist.size());

	for(unsigned i=0; i<size; i++) {
		ASSERT_EQ(vecSrc[i].getValue().rows(), vecDist[i].getValue().rows());
		ASSERT_EQ(vecSrc[i].getValue().cols(), vecDist[i].getValue().cols());
		for(unsigned j=0; j<3; j++) {
			for(unsigned k=0; k<4; k++) {
				EXPECT_NEAR(vecSrc[i].getValue()(j, k), vecDist[i].getValue()(j, k), 0.000001);
			}
		}
	}

}



TEST(Models, Polynom) {
	// Try to save and load a ts::Polynom

	ts::Polynom<float> srcModel(3, {3, 3});
	srcModel.save("tests/polynom.ts");


	ts::Polynom<float> dstModel(0, {0, 0});
	dstModel.load("tests/polynom.ts");


	// Compare values

	ASSERT_EQ(srcModel.coefficients.size(), dstModel.coefficients.size());

	for(unsigned i=0; i<srcModel.coefficients.size(); i++) {

		ASSERT_EQ(srcModel.coefficients[i].getValue().rows(), dstModel.coefficients[i].getValue().rows());
		ASSERT_EQ(srcModel.coefficients[i].getValue().cols(), dstModel.coefficients[i].getValue().cols());

		for(unsigned j=0; j<srcModel.coefficients[i].getValue().rows(); j++) {
			for(unsigned k=0; k<srcModel.coefficients[i].getValue().cols(); k++) {
				EXPECT_NEAR(srcModel.coefficients[i].getValue()(j, k), dstModel.coefficients[i].getValue()(j, k), 0.000001);
			}
		}

	}

}



TEST(Models, MultiLayerPerceptron) {
	// Try to save and load a ts::MultiLayerPerceptron

	ts::MultiLayerPerceptron<float> srcModel(2, {3});
	srcModel.save("tests/mlp.ts");

	ts::MultiLayerPerceptron<float> dstModel(0, {0});
	dstModel.load("tests/mlp.ts");


	// Compare values

	ASSERT_EQ(srcModel.weights.size(), dstModel.weights.size());

	for(unsigned i=0; i<srcModel.weights.size(); i++) {

		ASSERT_EQ(srcModel.weights[i].getValue().rows(), dstModel.weights[i].getValue().rows());
		ASSERT_EQ(srcModel.weights[i].getValue().cols(), dstModel.weights[i].getValue().cols());

		for(unsigned j=0; j<srcModel.weights[i].getValue().rows(); j++) {
			for(unsigned k=0; k<srcModel.weights[i].getValue().cols(); k++) {
				EXPECT_NEAR(srcModel.weights[i].getValue()(j, k), dstModel.weights[i].getValue()(j, k), 0.000001);
			}
		}

	}


	ASSERT_EQ(srcModel.biases.size(), dstModel.biases.size());

	for(unsigned i=0; i<srcModel.biases.size(); i++) {

		ASSERT_EQ(srcModel.biases[i].getValue().rows(), dstModel.biases[i].getValue().rows());
		ASSERT_EQ(srcModel.biases[i].getValue().cols(), dstModel.biases[i].getValue().cols());

		for(unsigned j=0; j<srcModel.biases[i].getValue().rows(); j++) {
			for(unsigned k=0; k<srcModel.biases[i].getValue().cols(); k++) {
				EXPECT_NEAR(srcModel.biases[i].getValue()(j, k), dstModel.biases[i].getValue()(j, k), 0.000001);
			}
		}

	}

}



TEST(Models, ConvolutionalNetwork) {
	// Try to save and load a ts::ConvolutionalNetwork

	ts::ConvolutionalNetwork<float> srcModel(
		{30, 10},
		ts::ChannelSplit::SPLIT_HOR, 3,
		{{3, 3, 2}},
		{{2,2}},
		{5, 6}
	);
	srcModel.save("tests/cnn.ts");

	ts::ConvolutionalNetwork<float> dstModel(
		{30, 10},
		ts::ChannelSplit::SPLIT_HOR, 3,
		{{3, 3, 2}},
		{{2,2}},
		{5, 6}
	);
	dstModel.load("tests/cnn.ts");


	// Compare outputs

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> randomInput;
	randomInput.setRandom(30, 10);

	// We need to create 2 tensors because the models use different wLists
	ts::Tensor<float> srcInput = ts::Tensor<float>(randomInput, &(srcModel.wList));
	ts::Tensor<float> dstInput = ts::Tensor<float>(randomInput, &(dstModel.wList));

	ts::Tensor<float> srcOutput = srcModel.compute(srcInput);
	ts::Tensor<float> dstOutput = dstModel.compute(dstInput);

	for(unsigned i=0; i<6; i++) {
		EXPECT_NEAR(srcOutput.getValue()(i, 0), dstOutput.getValue()(i, 0), 0.0001);
	}

}



int main(int argc, char **argv) {
	std::cout << "*** SERIALIZER TEST SUITE ***" << std::endl;

	srand(time(NULL));

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
