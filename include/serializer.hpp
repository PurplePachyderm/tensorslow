/*
* Set of helper functions to implement serialization and parsing of ts::Model
* derived classes.
*/

#pragma once

#include "autodiff.hpp"

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>


namespace ts::serializer {
	std::vector<std::string> split(std::string str, char delimeter);

	template <typename T> std::string serializeTensor(ts::Tensor<T> &tensor);
	template <typename T> Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> parseArray(
		std::ifstream in
	);

	template <typename T> std::string serializeTensorsVector(
		std::vector<ts::Tensor<T>> &tensorsVector
	);
}
