/*
* Set of helper functions to implement serialization and parsing of ts::Model
* derived classes.
*/

#include "../include/serializer.hpp"



// Serializes a ts::Tensor in a string.
// The output string has the following format:
// *ROWS*
// *COLS*
// *VAL*,*VAL*, *VAL*, ..., *VAL*
template <typename T>
std::string ts::serializeTensor(ts::Tensor<T> &tensor) {

	// Save array to string
	std::ostringstream stringStream;
	stringStream << tensor.getValue() << std::endl;

	std::string arrayString =  stringStream.str();


	// Because some extra spaces can be added by Eigen, we remove consecutive
	// spaces/linebreaks
	for(int i=arrayString.size()-2; i>=0; i--) {
		if(
			(arrayString[i] == ' ' || arrayString[i] == '\n') &&
			(arrayString[i+1] == ' ' || arrayString[i+1] == '\n')
		) {
			arrayString.erase(i, 1);
		}
	}

	// We might have a trailing space remaining at index 0 ...
	if(arrayString[0] == ' ') {
		arrayString.erase(0, 1);
	}

	// ... as well as a \n at last position
	if(arrayString[arrayString.size()-1] == '\n') {
		arrayString.erase(arrayString.size()-1, 1);
	}


	// Finally, remove spaces and linebreaks
	std::replace(arrayString.begin(), arrayString.end(), ' ', ',');
	std::replace(arrayString.begin(), arrayString.end(), '\n', ',');


	// Create out stream, return it as string
	std::ostringstream outStream;

	outStream << tensor.getValue().rows() << std::endl;
	outStream << tensor.getValue().cols() << std::endl;
	outStream << arrayString << std::endl;

	return outStream.str();
}



// Reads an ifstream starting at a serialized tensor, and parses it to a
// ts::Tensor
template <typename T>
ts::Tensor<T> ts::parseTensor(
	std::ifstream &in, ts::WengertList<T> * wList
) {

	std::string line;

	// Get rows
	std::getline(in, line);
	unsigned rows = std::stoi(line);
	std::cout << rows << std::endl;

	// Get cols
	std::getline(in, line);
	unsigned cols = std::stoi(line);

	// Get elements vector
	std::getline(in, line);
	std::cout << line << std::endl;
	std::vector<std::string> stringElements = ts::split(line, ',');


	// Initialize Eigen::Array
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> array;
	array.resize(rows, cols);

	for(unsigned i=0; i<rows; i++) {
		for(unsigned j=0; j<cols; j++) {
			array(i, j) = std::stof(stringElements[i * cols + j]);
		}
	}


	return ts::Tensor<T>(array, wList);
}



// Serializes a vector of ts::Tensor using the ts::serializeTensor
// method. The outputted string has the following format :
// *N == VECTOR SIZE*
// *TENSOR 1 (3 lines)*
// *TENSOR 2 (3 lines)*
// ...
// *TENSOR N (3 lines)*
template <typename T>
std::string ts::serializeTensorsVector(
	std::vector<ts::Tensor<T>> &tensorsVector
) {

	std::ostringstream stringStream;

	stringStream << tensorsVector.size() << std::endl;

	for(unsigned i=0; i<tensorsVector.size(); i++) {
		stringStream << ts::serializeTensor(tensorsVector[i]) << std::endl;
	}

	return stringStream.str();
}



// Reads an ifstream starting at a serialized tensors vector, and parses it to
// a std::vector<ts::Tensor>
template <typename T>
std::vector<ts::Tensor<T>> ts::parseTensorsVector(
	std::ifstream &in, ts::WengertList<T> * wList
) {
	std::vector<ts::Tensor<T>> vector = {};

	std::string line;

	// Get tensor size
	std::getline(in, line);
	unsigned size = std::stoi(line);

	for(unsigned i=0; i<size; i++) {
		vector.push_back(ts::parseTensor(in, wList));
	}

	return vector;
}
