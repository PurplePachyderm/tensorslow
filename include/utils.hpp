/*
* Misc utility functions to be reused in main parts of the library
* NOTE For now, this has only been created to avoid double definition of split
*/

#include <vector>
#include <string>
#include <sstream>

namespace ts {
	std::vector<std::string> split(std::string str, char delimeter);
}
