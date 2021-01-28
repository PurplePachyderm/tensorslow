/*
* Misc utility functions to be reused in main parts of the library
* NOTE For now, this has only been created to avoid double definition of split
*/

#include "../include/utils.hpp"



// Splits a string by a char delimiter and returns substrings in a vector
std::vector<std::string> ts::split(std::string str, char delimeter){

	std::stringstream ss(str);
	std::string item;
	std::vector<std::string> splitted;

	while (std::getline(ss, item, delimeter)) {
		splitted.push_back(item);
	}

	return splitted;
}



// Read / write 2D unsigned vector

std::string ts::serializeUnsignedVec2D(
	std::vector<std::vector<unsigned>> &vec2d
) {
	std::ostringstream stringStream;

	stringStream << vec2d.size() << std::endl;

	for(unsigned i=0; i<vec2d.size(); i++) {
		stringStream << vec2d[i].size() << std::endl;
		for(unsigned j=0; j<vec2d[i].size(); j++) {
			stringStream << vec2d[i][j] << std::endl;
		}
	}

	return stringStream.str();
}



std::vector<std::vector<unsigned>> ts::parseUnsignedVec2D(
	std::ifstream &in
) {

	std::vector<std::vector<unsigned>> vec2d = {};

	std::string line;

	// Get vec2dsize size
	std::getline(in, line);
	unsigned size2d = std::stoi(line);

	for(unsigned i=0; i<size2d; i++) {
		vec2d.push_back({});

		std::getline(in, line);
		unsigned size1d = std::stoi(line);

		for(unsigned j=0; j<size1d; j++) {
			std::getline(in, line);
			unsigned val = std::stoi(line);
			vec2d[i].push_back(val);
		}
	}

	return vec2d;
}
