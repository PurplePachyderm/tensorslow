/*
* Implementation for AD of convolution-related operations, that will typically
* be used in a CNN.
*/

#include "../include/convolution.hpp"



	// Convolution operation on Eigen arrays
	// (LEGACY, for benchmarking purpose only)

template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::convArray(
	const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &mat,
	const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &ker
) {

	// Make sure kernel is smaller
	if(
		mat.rows() < ker.rows() ||
		mat.cols() < ker.cols()
	) {
		return Eigen::Array<T, 0, 0>();
	}

	unsigned newRows = mat.rows() - ker.rows() + 1;
	unsigned newCols = mat.cols() - ker.cols() + 1;


	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> res;
	res.resize(newRows, newCols);

	// #pragma omp parallel for collapse(2)
	for(unsigned i=0; i<newCols; i++) {
		for(unsigned j=0; j<newRows; j++) {
			// Compute one element of feature map
			res(j, i) =
			(mat.block(j, i, ker.rows(), ker.cols()) * ker).sum();
		}
	}

	return res;
}



	// Convolution
	// (LEGACY, for benchmarking purpose only)

template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::ConvolutionNode<T>::incrementGradient(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
		unsigned &j
) {

	// Used in the  ts::Tensor::grad() method. Computes the increment of a derivative
	// for a convolution operation.

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> increment;

	// Matrices are already prepared at this stage, so we only need to put the
	// operands in the correct order for convolution.

	if(
		childDerivative.rows() > this->values[j].rows() &&
		childDerivative.rows() > this->values[j].cols()
	) {
		increment = ts::convArray(childDerivative, this->values[j]);
	} else {
		increment = ts::convArray(this->values[j], childDerivative);
	}

	return increment;
}



template <typename T>
ts::Tensor<T> ts::convolution(const ts::Tensor<T> &mat, const ts::Tensor<T> &ker) {
	// Convolution operation
	// Resulting matrix is of size : (mat.x - ker.x + 1, mat.y - ker.y + 1)
	// (where mat.x >= ker.x and mat.y >= mat.y)

	if(
		mat.wList != ker.wList ||
		mat.value.rows() < ker.value.rows() ||
		mat.value.cols() < ker.value.cols()
	) {
		return ts::Tensor<T>(Eigen::Array<T, 0, 0>(), NULL);
	}

	// The gradient will have to be computed for a scalar
	mat.wList->elementWiseOnly = false;


	// Compute res
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> res = ts::convArray(
		mat.value, ker.value
	);

	// Init dMat matrix (for matrix partial derivative)
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> dMat;
	dMat.setZero(
		2 * res.rows() + ker.value.rows() - 2,
		2 * res.cols() + ker.value.cols() - 2
	);

	dMat.block(
		res.rows() - 1,
		res.cols() - 1,
		ker.value.rows(), ker.value.cols()
	) = ker.value.rowwise().reverse().colwise().reverse();

	std::shared_ptr<ts::Node<T>> nodePtr (
		new ts::ConvolutionNode<T>(
			{res.rows(), res.cols()},
			dMat, mat.index,
			mat.value, ker.index
		)
	);

	return ts::Tensor<T>(res, mat.wList, nodePtr);
}



	// Max pooling

template <typename T>
ts::PoolingNode<T>::PoolingNode(
	std::vector<long> shape,
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> xVal, int xDep,
	std::vector<unsigned> newPool
) {

	// PoolingNode specific constructor to store the size of pools
	// (this allows us to easily upscale the matrix in grad computation)

	this->rows = shape[0];
	this->cols = shape[1];

	this->values =  {xVal};	// [da/dx]
	this->dependencies =  {xDep};

	pool = newPool;
}



template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::PoolingNode<T>::incrementGradient(
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
	unsigned &j
) {

	// Used in the  ts::Tensor::grad() method. Computes the increment of a derivative
	// for a max pooling / downsample operation.


	// Upsample matrix of child derivative to match original size

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> upsample;
	upsample.setZero(this->values[j].rows(), this->values[j].cols());


	// Affect coefficients of childDerivative to upsample pools by filling each
	// pool with the corresponding value

	// #pragma omp parallel for collapse(2)
	for(unsigned i=0; i<childDerivative.cols(); i++) {
		for(unsigned j=0; j<childDerivative.rows(); j++) {

			// Fill one pool with one value
			for(unsigned k=0; k<pool[1]; k++) {
				for(unsigned l=0; l<pool[0]; l++) {
					upsample(j * pool[0] + l, i * pool[1] + k) =
					childDerivative(j, i);
				}
			}

		}
	}


	// Compute & return element-wise product with the this->values
	// (since this->values is 0/1-flled, we will only get the coefficients in
	// the desired positions, and 0 anywhere else)
	return upsample * this->values[j];
}



template <typename T>
ts::Tensor<T> ts::maxPooling(const ts::Tensor<T> &x, std::vector<unsigned> pool) {
	// Max pooling operation : we keep only the biggest element in each pool
	// in order to reduce the size of a matrix
	// Resulting matrix is of size : (mat.x / pool.x, mat.y / pool.y)
	// (where mat.x >= pool.x and mat.y >= pool.y)

	if(pool.size() != 2) {
		return ts::Tensor<T>(Eigen::Array<T, 0, 0>(), NULL);
	}

	if(
		x.value.rows() % pool[0] != 0 ||
		x.value.cols() % pool[1] != 0
	) {
		return ts::Tensor<T>(Eigen::Array<T, 0, 0>(), NULL);
	}

	// The gradient will have to be computed for a scalar
	x.wList->elementWiseOnly = false;


	// Init result
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> res;
	res.setZero(x.value.rows() / pool[0], x.value.cols() / pool[1]);


	// Init dx
	// (dx is 1 for each max element, 0 elsewhere)
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> dx;
	dx.setZero(x.value.rows(), x.value.cols());


	unsigned xMax, yMax;
	T maxVal;


	// Compute both pooled matrix (res) and dx
	// #pragma omp parallel for collapse(2)
	for(unsigned i=0; i<res.cols(); i++) {
		for(unsigned j=0; j<res.rows(); j++) {

			// Get index of pool's max element
			// (for now it seems the best way is to manually iterate over
			// elements)

			xMax = j * pool[0];
			yMax = i * pool[1];
			maxVal = x.value(j * pool[0], i * pool[1]);

			for(unsigned k=0; k<pool[1]; k++) {
				for(unsigned l=0; l<pool[0]; l++) {

					if(x.value(j * pool[1] + l, i * pool[0] + k) > maxVal) {
						maxVal = x.value(j * pool[1] + l, i * pool[0] + k);
						xMax = j * pool[1] + l;
						yMax = i * pool[0] + k;
					}

				}
			}

			// Assigning values for result and derivative
			res(j, i) = maxVal;
			dx(xMax, yMax) = 1.0;

		}
	}


	std::shared_ptr<ts::Node<T>> nodePtr (
		new ts::PoolingNode<T>(
			{res.rows(), res.cols()},
			dx, x.index,
			pool
		)
	);

	return ts::Tensor<T>(res, x.wList, nodePtr);
}



	// Splitting

template <typename T>
ts::SplitNode<T>::SplitNode(
	std::vector<long> shape,
	int xDep,
	std::vector<long> originalShape,
	ChannelSplit newSplitDirection,
	unsigned newPosition
) {
	// SplitNode specific constructor to store the split direction

	this->dependencies =  {xDep};

	// New tensor shape (dimension of split matrices)
	this->rows = shape[0];
	this->cols = shape[1];

	// Original matrix shape
	originalRows = originalShape[0];
	originalCols = originalShape[1];


	splitDirection = newSplitDirection;
	position = newPosition;
}



template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::SplitNode<T>::incrementGradient(
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
	unsigned &j
) {

	// Used in the  ts::Tensor::grad() method. Computes the increment of a derivative
	// for a matrix split.

	// childDerivative is one of the resulting matrices. We will reconstruct
	// the partial derivative with regard to this considered matrix in order to
	// compute the increment. Index of the corresponding matrix is given by j.

	// Shape of base matrix derivative (initally zero filled)
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> increment;
	increment.setZero(originalRows, originalCols);

	// Affect childDerivative values to correct positions, according to
	// split direction & matrix index (j)
	if(splitDirection == ChannelSplit::SPLIT_VERT) {
		increment.block(0, position * this->cols, this->rows, this->cols) =
		childDerivative;
	}

	else if(splitDirection == ChannelSplit::SPLIT_HOR) {
		increment.block(position * this->rows, 0, this->rows, this->cols) =
		childDerivative;
	}

	return increment;
}



template <typename T>
std::vector<ts::Tensor<T>> ts::split(
	const ts::Tensor<T> &x,
	ChannelSplit channelSplit,
	unsigned nInputChannels
) {

	// The gradient will have to be computed for a scalar
	x.wList->elementWiseOnly = false;

	std::vector<ts::Tensor<T>> matrices = {};

	if(channelSplit == ChannelSplit::SPLIT_HOR) {
		unsigned channelSize = x.value.rows() / nInputChannels;

		for(unsigned i=0; i<nInputChannels; i++) {

			// Get matrix form of block
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> tmp =
			x.value.block(
				i * channelSize, 0, channelSize, x.value.cols()
			);

			// Create associated Tensor
			std::shared_ptr<ts::Node<T>> nodePtr (
				new ts::SplitNode<T>(
					{channelSize, x.value.cols()},
					x.index,
					{x.value.rows(), x.value.cols()},
					channelSplit,
					i
				)
			);

			matrices.push_back(ts::Tensor<T>(tmp, x.wList, nodePtr));
		}
	}

	if(channelSplit == ChannelSplit::SPLIT_VERT) {
		unsigned channelSize = x.value.cols() / nInputChannels;

		for(unsigned i=0; i<nInputChannels; i++) {

			// Get matrix form of block
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> tmp =
			x.value.block(
				0, i * channelSize, x.value.rows(), channelSize
			);

			// Create associated Tensor
			std::shared_ptr<ts::Node<T>> nodePtr (
				new ts::SplitNode<T>(
					{x.value.rows(), channelSize},
					x.index,
					{x.value.rows(), x.value.cols()},
					channelSplit,
					i
				)
			);

			matrices.push_back(ts::Tensor<T>(tmp, x.wList, nodePtr));

		}
	}

	if(channelSplit == ChannelSplit::NOSPLIT) {
		matrices.push_back(x);
	}

	return matrices;
}



	// Vertical concatenation

template <typename T>
ts::VertCatNode<T>::VertCatNode(
	std::vector<long> shape,
	std::vector<int> newDependencies,
	std::vector<long> newHeights
) {

	// VertCatNode specific constructor to store the height of first matrix
	// This way we can copy correct elements in inrementGradient

	// New tensor shape (vector)
	this->rows = shape[0];
	this->cols = shape[1];

	this->dependencies =  newDependencies;

	// Height of the first matrix
	heights = newHeights;
}



template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::VertCatNode<T>::incrementGradient(
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
	unsigned &j
) {

	// Used in the  ts::Tensor::grad() method. Computes the increment of a derivative
	// for a matrix flattening.

	// childDerivative is a flattened vector. We need to convert it back to a
	// matrix with the dimensions of the original matrix.

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> mat;
	mat.resize(heights[j+1] - heights[j], childDerivative.cols());

	mat = childDerivative.block(
		heights[j], 0,
		heights[j+1] - heights[j], childDerivative.cols()
	);


	return mat;
}



template <typename T>
ts::Tensor<T> ts::vertCat(const std::vector<ts::Tensor<T>> &x) {
	// Vertical concatenation operation
	// x[i] will be under x[i-1]

	if(x.size() == 0) {
		return ts::Tensor<T>(Eigen::Array<T, 0, 0>(), NULL);
	}


	// The gradient will have to be computed for a scalar
	x[0].wList->elementWiseOnly = false;


	// Compute size of resulting matrix, and storing each input matrix position
	// We will also make sure that all matrices have the same width / wList

	std::vector<long> heights = {0}; // Values are cumulative starting heights
	long height = 0;
	long width = x[0].value.cols();
	std::vector<int> dependencies = {};

	for(unsigned i=0; i<x.size(); i++) {

		if(x[i].value.cols() != width || x[i].wList != x[0].wList) {
			return ts::Tensor<T>(Eigen::Array<T, 0, 0>(), NULL);
		}

		height += x[i].value.rows();
		heights.push_back(height);
		dependencies.push_back(x[i].index);

	}


	// Set res vector
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> res;
	res.resize(height, width);

	for(unsigned i=0; i<x.size(); i++) {
		res.block(heights[i], 0, heights[i+1] - heights[i], width) = x[i].value;
	}


	// Return
	std::shared_ptr<ts::Node<T>> nodePtr (
		new ts::VertCatNode<T>(
			{res.rows(), res.cols()},
			dependencies,
			heights
		)
	);

	return ts::Tensor<T>(res, x[0].wList, nodePtr);
}



	// Flattening

template <typename T>
ts::FlatteningNode<T>::FlatteningNode(
	std::vector<long> shape,
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> xVal, int xDep,
	std::vector<long> newSize
) {

	// FlatteningNode specific constructor to store the size of original matrix
	// (this allows us to easily rescale the flattened vector in grad
	// computation)

	// New tensor shape (vector)
	this->rows = shape[0];
	this->cols = shape[1];

	this->values =  {xVal};	// [da/dx]
	this->dependencies =  {xDep};

	// Original matrix size
	size = newSize;
}



template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::FlatteningNode<T>::incrementGradient(
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
	unsigned &j
) {

	// Used in the  ts::Tensor::grad() method. Computes the increment of a derivative
	// for a matrix flattening.

	// childDerivative is a flattened vector. We need to convert it back to a
	// matrix with the dimensions of the original matrix.

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> mat;
	mat.setZero(size[0], size[1]);

	// #pragma omp parallel for collapse(2)
	for(unsigned i=0; i<size[1]; i++) {
		for(unsigned j=0; j<size[0]; j++) {
			mat(j, i) = childDerivative(j * size[1] + i, 0);
		}
	}

	return mat;
}



template <typename T>
ts::Tensor<T> ts::flattening(const ts::Tensor<T> &x) {
	// Flattening operation to convert matrix to vector
	// A x matrix of size m*n becomes the following vector :
	// x(1,1), ..., x(1, n), x(2,1), ..., x(m, n)
	// (the resulting size is (m*n, 1)

	// The gradient will have to be computed for a scalar
	x.wList->elementWiseOnly = false;


	// Set res vector
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> res = x.value;
	res = Eigen::Map<Eigen::Array<T, -1, 1>>(
		res.data(), res.cols() * res.rows()
	);


	// Set dx matrix
	// It should be 1-filled since we're keeping all values of x in res, but
	// storing the full matrix would not be memory-efficient
	Eigen::Array<T, 0, 0> dx;


	// Return
	std::shared_ptr<ts::Node<T>> nodePtr (
		new ts::FlatteningNode<T>(
			{res.rows(), res.cols()},
			dx, x.index,
			{x.value.rows(), x.value.cols()}
		)
	);

	return ts::Tensor<T>(res, x.wList, nodePtr);
}



	// Im2col

template <typename T>
ts::Im2ColNode<T>::Im2ColNode(
	std::vector<long> shape,
	std::vector<int> newDependencies,
	std::vector<long> newKernelDim,
	std::vector<long> newMatrixDim,
	unsigned newNChannels
) {
	// New tensor shape (vector)
	this->rows = shape[0];
	this->cols = shape[1];

	this->dependencies =  newDependencies;

	// Original matrix size
	kernelDim = newKernelDim;
	matrixDim = newMatrixDim;
	nChannels = newNChannels;
}



template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::Im2ColNode<T>::incrementGradient(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
		unsigned &j
) {

	// Used in the  ts::Tensor::grad() method. Computes the increment of a derivative
	// for a im2col operation.

	// childDerivative has the shape of the final matrix.
	// The increment will have the shape of one input matrix (this method will
	// be called once for each channel)

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> mat;
	mat.setZero(matrixDim[0], matrixDim[1]);

	// This matrix will be converted back to "normal" shape
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> im2colMat = childDerivative.block(
		j * kernelDim[0] * kernelDim[1], 0,
		kernelDim[0] * kernelDim[1], childDerivative.cols()
	);

	// #pragma omp parallel for collapse(2)
	for(unsigned i=0; i<im2colMat.cols(); i++) {
		// Each column is a col-major flattened submatrix
		for(unsigned j=0; j<im2colMat.rows(); j++) {
			// Get top left coords of submatrix
			int submatTopX = i / (matrixDim[0] - kernelDim[0] + 1);
			int submatTopY = i % (matrixDim[0] - kernelDim[0] + 1);

			// Get coords in submatrix
			int submatX = j / kernelDim[1];
			int submatY = j % kernelDim[1];

			// Add derivative to coords in original matrix
			mat(submatTopX + submatX, submatTopY + submatY) =
			mat(submatTopX + submatX, submatTopY + submatY) + im2colMat(j, i);

		}
	}

	return mat;
}



template <typename T>
ts::Tensor<T> ts::im2col(
	const std::vector<ts::Tensor<T>> &x,
	std::vector<unsigned> kernelDim
) {
	// Turns a tensor vector into a single im2col matrix
	// Using a kernels matrix, one entire conv layer could be computed in
	// only one matrix product

	std::vector<int> dependencies = {};

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> res;
	res.resize(
		kernelDim[0] * kernelDim[1] * x.size(),
		(x[0].value.rows() - kernelDim[0] + 1) * (x[0].value.cols() - kernelDim[1] + 1)
	);

	for(unsigned i=0; i<x.size(); i++) {
		// #pragma omp parallel for collapse(2)
		for(unsigned j=0; j<x[i].value.cols() - kernelDim[0] + 1; j++) {
			for(unsigned k=0; k<x[i].value.rows() - kernelDim[1] + 1; k++) {

				Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> tmp =
				x[i].value.block(k, j, kernelDim[0], kernelDim[1]);

				Eigen::Map<Eigen::Array<T, -1, 1>> map =
				Eigen::Map<Eigen::Array<T, -1, 1>>(
					tmp.data(), tmp.cols() * tmp.rows()
				);

				res.block(i * kernelDim[0] * kernelDim[1], k * (x[i].value.rows() - kernelDim[0] + 1)  + j, kernelDim[0] * kernelDim[1], 1)
				= map;

			}
		}

		dependencies.push_back(x[i].index);
	}


	// Return
	std::shared_ptr<ts::Node<T>> nodePtr (
		new ts::Im2ColNode<T>(
			{res.rows(), res.cols()},
			dependencies,
			{kernelDim[0], kernelDim[1]},
			{x[0].value.rows(), x[0].value.cols()},
			x.size()
		)
	);

	return ts::Tensor<T>(res, x[0].wList, nodePtr);
}



	// Col2im

template <typename T>
ts::Col2ImNode<T>::Col2ImNode(
	std::vector<long> shape,
	int xDep,
	unsigned newPosition,
	long newNChannels
) {
	// New tensor shape (vector)
	this->rows = shape[0];
	this->cols = shape[1];

	this->dependencies =  {xDep};

	// Original matrix size
	position = newPosition;
	nChannels = newNChannels;
}



template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::Col2ImNode<T>::incrementGradient(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &childDerivative,
		unsigned &j
) {

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> flat = childDerivative;
	flat = Eigen::Map<Eigen::Array<T, 1, -1>>(
		flat.data(), flat.cols() * flat.rows()
	);

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> res;
	res.setZero(nChannels, flat.cols());
	res.block(position, 0, 1, flat.cols()) = flat;

	return res;
}



template <typename T>
std::vector<ts::Tensor<T>> ts::col2im(
	const ts::Tensor<T> &x,
	std::vector<unsigned> outputDim
) {
	// Turns an im2col matrix into a channels vector
	// The output can be reused in another im2col, or
	// flattened before dense layers.

	std::vector<ts::Tensor<T>> res = {};

	// Each line contains some channel's coefficients in row-major order
	for(unsigned i=0; i<x.value.rows(); i++) {
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> tmp =
		x.value.block(i, 0, 1, x.value.cols());

		tmp.resize(outputDim[0], outputDim[1]);


		// COnvert it back to matrix form
		std::shared_ptr<ts::Node<T>> nodePtr (
			new ts::Col2ImNode<T>(
				{tmp.rows(), tmp.cols()},
				x.index,
				i,
				x.value.rows()
			)
		);

		res.push_back(ts::Tensor<T>(tmp, x.wList, nodePtr));
	}

	return res;
}
