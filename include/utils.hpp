/*
* Misc utility functions to be reused in main parts of the library
* NOTE For now, this has only been created to avoid double definition of split
*/

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#define BARWIDTH 30

namespace ts {
	std::vector<std::string> split(std::string str, char delimeter);

	std::string serializeUnsignedVec2D(
		std::vector<std::vector<unsigned>> &vec2d
	);

	std::vector<std::vector<unsigned>> parseUnsignedVec2D(
		std::ifstream &in
	);

	void progressBar(unsigned current, unsigned max);
}
