/*
* This is the prediction phase of the R-CNN example. We'll import the pre-traines
* CNN (run rcnn_train before) and use it in a R-CNN to detext vehicles in
* images from the Traffic_Net dataset.
*
* NOTE: Run the examples/get-trafficnet.sh script before this example to download
* the Traffic_Net dataset. Both the script and this example must be run from the
* root of the repository.
*/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <CImg.h>

#include "../include/tensorslow.h"

#define IMAGE_WIDTH 32
// Images are stored in row-major order with each color component separated
#define IMAGE_HEIGHT 96
#define N_CLASSES 10
#define DETECTION_THRESHOLD 0.999
#define STRIDE_X 5
#define STRIDE_Y 5



Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> readImage(
	std::string filepath
) {

	// From its filepath, reads a .jpg image and store its RGB components in
	// a Eigen::Array

	// Read file
	cimg_library::CImg<unsigned char> src(filepath.c_str());
	unsigned width = src.width();
	unsigned height = src.height();

	// Initialize Eigen array
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> res;
	res.resize(height * 3, width);

	// Fill Eigen array (with vertical channel split)
	for(unsigned i=0; i<height; i++) {
		for(unsigned j=0; j<width; j++) {
			// R
			res(i, j) = (float) src(j, i, 0, 0) / 255.0f;
			// G
			res(height + i, j) = (float) src(j, i, 0, 1) / 255.0f;
			// B
			res(2 * height + i, j) = (float) src(j, i, 0, 2) / 255.0f;
		}
	}

	return res;
}



int writeImage(
	std::string filepath,
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> array
) {

	// Writes the img Eigen::Array as a .jpg image in filepath

	unsigned width = array.cols();
	unsigned height = array.rows() / 3;

	// Create Cimg object
	cimg_library::CImg<unsigned char> img(
		width, height,
		1, 3
	);

	for(unsigned i=0; i<height; i++) {
		for(unsigned j=0; j<width; j++) {

			// R
			img(j, i, 0, 0) = (unsigned char) (array(height + i, j) * 255.0f);
			// G
			img(j, i, 0, 1) = (unsigned char) (array(height + i, j) * 255.0f);
			// B
			img(j, i, 0, 2) = (unsigned char) (array(2 * height + i, j) * 255.0f);
		}
	}

	img.save(filepath.c_str());

	return 0;
}



Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> rcnn(
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> img,
	ts::ConvolutionalNetwork<float> &cnn,
	std::vector<unsigned> strides
) {

	// Performs a R-CNN prediction on the img with the CNN model.
	// This will return the probability matrix for the image (each cell
	// corresponds to the probability of having a vehicle in the
	// region). The strides parameter defines the shift between each
	// CNN forward pass. This will be help to reduce calculations despite
	// our naive approach.

	unsigned width = img.cols();
	unsigned height = img.rows() / 3;

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> res;

	// Get number of CNN forward passes to perform in order to resize res
	unsigned nxPass = (height - IMAGE_WIDTH) / strides[0];
	unsigned nyPass = (width - IMAGE_WIDTH) / strides[1];

	res.resize(nxPass, nyPass);


	// Perform CNN forward passes over img

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> submat;
	submat.resize(IMAGE_HEIGHT, IMAGE_WIDTH);

	for(unsigned i=0; i<nxPass; i++) {
		for(unsigned j=0; j<nyPass; j++) {
			// Get coords of area's top-left corner
			unsigned x = i * strides[0];
			unsigned y = j * strides[1];

			// Prepare submatrix with 3 channels
			// R
			submat.block(0, 0, IMAGE_WIDTH, IMAGE_WIDTH) =
			img.block(x, y, IMAGE_WIDTH, IMAGE_WIDTH);
			// G
			submat.block(IMAGE_WIDTH, 0, IMAGE_WIDTH, IMAGE_WIDTH) =
			img.block(height + x, y, IMAGE_WIDTH, IMAGE_WIDTH);
			// B
			submat.block(2 * IMAGE_WIDTH, 0, IMAGE_WIDTH, IMAGE_WIDTH) =
			img.block(2 * height + x, y, IMAGE_WIDTH, IMAGE_WIDTH);

			// Create tensor & perform forward pass
			ts::Tensor<float> base = ts::Tensor<float>(
				submat,
				&(cnn.wList)
			);

			ts::Tensor<float> prob = cnn.compute(base);
			cnn.wList.reset();

			res(i, j) = prob.getValue()(0, 0);
		}
	}

	return res;
}



Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> drawBoundingBoxes(
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> img,
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> probabilityMatrix,
	std::vector<unsigned> strides
) {

	// Given an image and its probability matrix, draws the bounding boxes
	// of detected objects and returns the resulting Eigen::Array.
	// TODO The current (lazy) approach draws a box for each region with a value
	// over detection threshold. It might be a good idea to gather overlapping
	// boxes in a single one in order to make output image cleaner and avoid
	// duplicate　detection.


	unsigned channelHeight = img.rows() / 3;

	for(unsigned i=0; i<probabilityMatrix.rows(); i++) {
		for(unsigned j=0; j<probabilityMatrix.cols(); j++) {
			if(probabilityMatrix(i, j) > DETECTION_THRESHOLD) {
				// Get box's top left coordinates in actual image
				unsigned topLeftX = i * strides[0];
				unsigned topLeftY = j * strides[1];

				// Deduce other box bounds
				unsigned bottomLeftX = topLeftX + IMAGE_WIDTH;
				unsigned topRightY = topLeftY + IMAGE_WIDTH;

				// Draw left line
				for(unsigned i=0; i<IMAGE_WIDTH; i++) {
					if(topLeftX + i < img.rows() && topLeftY < img.cols()) {
						img(topLeftX + i, topLeftY) = 1.0f;
						img(channelHeight + topLeftX + i, topLeftY) = 0.0f;
						img(channelHeight * 2 + topLeftX + i, topLeftY) = 0.0f;
					}
				}

				// Draw top line
				for(unsigned i=0; i<IMAGE_WIDTH; i++) {
					if(topLeftX < img.rows() && topLeftY + i < img.cols()) {
						img(topLeftX, topLeftY + i) = 1.0f;
						img(channelHeight + topLeftX, topLeftY + i) = 0.0f;
						img(channelHeight * 2 + topLeftX, topLeftY + i) = 0.0f;
					}
				}

				// Draw right line
				for(unsigned i=0; i<IMAGE_WIDTH; i++) {
					if(topLeftX + i < img.rows() && topRightY < img.cols()) {
						img(topLeftX + i, topRightY) = 1.0f;
						img(channelHeight + topLeftX + i, topRightY) = 0.0f;
						img(channelHeight * 2 + topLeftX + i, topRightY) = 0.0f;
					}
				}

				// Draw bottom line
				for(unsigned i=0; i<IMAGE_WIDTH; i++) {
					if(bottomLeftX < img.rows() && topLeftY + i < img.cols()) {
						img(bottomLeftX, topLeftY + i) = 1.0f;
						img(channelHeight + bottomLeftX, topLeftY + i) = 0.0f;
						img(channelHeight * 2 + bottomLeftX, topLeftY + i) = 0.0f;
					}
				}
			}
		}
	}

	return img;
}


	// Main

int main(void) {

		// Import the pre-trained CNN

	ts::ConvolutionalNetwork<float> model(
		// Input
		{IMAGE_HEIGHT, IMAGE_WIDTH},

		// Number of channels for input (3 for RGB)
		ts::ChannelSplit::SPLIT_HOR, 3,

		// Convolution / pooling
		{{3, 3, 32}, {5, 5, 16}},
		{{0,0}, {2, 2}},

		// Dense layers (with output vector & not including first layer)
		{128, 64, N_CLASSES, 1}
	);
	model.load("examples/rcnn.ts");
	// model.toggleGlobalOptimize(true);
	std::cout << "Imported the CNN ..." << std::endl;


		// Define images to load from dataset

	std::vector<std::string> imgPath = {
		"examples/trafficnet/train/sparse_traffic/images_214.jpg",
		"examples/trafficnet/train/sparse_traffic/images_215.jpg",
		"examples/trafficnet/train/sparse_traffic/images_216.jpg",
		"examples/trafficnet/train/sparse_traffic/images_217.jpg",
		"examples/trafficnet/train/sparse_traffic/images_218.jpg",
		"examples/trafficnet/train/sparse_traffic/images_219.jpg",
		"examples/trafficnet/train/sparse_traffic/images_224.jpg",
		"examples/trafficnet/train/sparse_traffic/images_228.jpg",
		"examples/trafficnet/train/sparse_traffic/images_229.jpg",
		"examples/trafficnet/train/sparse_traffic/images_232.jpg",
		"examples/trafficnet/train/sparse_traffic/images_262.jpg",
		"examples/trafficnet/train/sparse_traffic/images_271.jpg",
		"examples/trafficnet/train/sparse_traffic/images_282.jpg",
		"examples/trafficnet/train/sparse_traffic/images_311.jpg",
		"examples/trafficnet/train/sparse_traffic/images_315.jpg",
		"examples/trafficnet/train/sparse_traffic/images_323.jpg",
		"examples/trafficnet/train/sparse_traffic/images_330.jpg",
		"examples/trafficnet/train/sparse_traffic/images_340.jpg",
		"examples/trafficnet/train/sparse_traffic/images_350.jpg",
		"examples/trafficnet/train/sparse_traffic/images_387.jpg"
	};


		// Process each image

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> img;
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> probabilityMatrix;

	for(unsigned i=0; i<imgPath.size(); i++) {
		std::cout << "Generating image " << i + 1 << "/" << imgPath.size() << std::endl;

		img = readImage(imgPath[i]);

		probabilityMatrix = rcnn(img, model, {STRIDE_X, STRIDE_Y});

		img = drawBoundingBoxes(img, probabilityMatrix, {STRIDE_X, STRIDE_Y});

		writeImage(
			"examples/trafficnet/results/" + std::to_string(i+1) + ".jpg",
			img
		);
	}

	return 0;
}
