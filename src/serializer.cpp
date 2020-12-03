/*
* Set of helper functions to implement serialization and parsing of ts::Model
* derived classes.
*/

#include "../include/serializer.hpp"



// Splits a string by a char delimiter and returns substrings in a vector
std::vector<std::string> ts::serializer::split(std::string str, char delimeter) {

	std::stringstream ss(str);
	std::string item;
	std::vector<std::string> splitted;

	while (std::getline(ss, item, delimeter)) {
		splitted.push_back(item);
	}

	return splitted;
}



// Serializes the Eigen::Array contained in a ts::Tensor.
// The output string has the following format:
// *ROWS*
// *COLS*
// *VAL*,*VAL*, *VAL*, ..., *VAL*
template <typename T>
std::string ts::serializer::serializeTensor(ts::Tensor<T> &tensor) {

	//Write rows and cols


	// Save array to string
	std::ostringstream stringStream;
	stringStream << tensor.getValue();

	std::string str =  stringStream.str();


	// Because some extra spaces can be added by Eigen, we remove consecutive
	// spaces/linebreaks
	for(unsigned i=str.size()-2; i>0; i--) {
		if(
			(str[i] == ' ' || str[i] == '\n') &&
			(str[i+1] == ' ' || str[i+1] == '\n')
		) {
			str.erase(i, 1);
		}
	}

	// We might have a trailing space remaining at index 0
	if(str[0] == ' ') {
		str.erase(0, 1);
	}


	// Finally, remove spaces and linebreaks
	std::replace(str.begin(), str.end(), ' ', ',');
	std::replace(str.begin(), str.end(), '\n', ',');

	return str;

}



// Reads an ifstream starting at a serialized tensor, and parses it to a
// ts::Tensor
template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ts::serializer::parseTensor(
	std::ifstream in, ts::WengertList<T> * wList
) {

	std::string line;

	// Get rows
	std::getline(in, line);
	int rows = std::stoi(line);

	// Get cols
	std::getline(in, line);
	int cols = std::stoi(line);

	// Get elements vector
	std::getline(in, line);
	std::vector<std::string> stringElements = ts::serializer::split(line, ',');


	// Initialize Eigen::Array
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> array;
	array.resize(rows, cols);

	for(unsigned i=0; i<rows; i++) {
		for(unsigned j=0; j<cols; j++) {
			array(i, j) = std::stof(stringElements[i * rows + j]);
		}
	}

	return ts::Tensor<T>(array, wList);
}



// Serializes a vector of ts::Tensor using the ts::serializer::serializeTensor
// method. The outputted string has the following format :
// *N == VECTOR SIZE*
// *TENSOR 1 (3 lines)*
// *TENSOR 2 (3 lines)*
// ...
// *TENSOR N (3 lines)*
template <typename T>
std::string serializeTensorsVector(
	std::vector<ts::Tensor<T>> &tensorsVector
) {
	std::string str;
	str << tensorsVector.size() << std::endl;

	for(unsigned i=0; i<tensorsVector.size(); i++) {
		str << ts::serializer::serializeTensor(tensorsVector[i]) << std::endl;
	}

	return str;
}



// Reads an ifstream starting at a serialized tensors vector, and parses it to
// a std::vector<ts::Tensor>
template <typename T>
std::string parseTensorsVector(
	std::ifstream in, ts::WengertList<T> * wList
) {
	std::vector<ts::Tensor<T>> vector = {};

	std::string line;

	// Get tensor size
	std::getline(in, line);
	unsigned size = std::stoi(line);

	for(unsigned i=0; i<size; i++) {
		vector.push_bask(parseVector(in, wList));
	}

	return vector;
}
