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
