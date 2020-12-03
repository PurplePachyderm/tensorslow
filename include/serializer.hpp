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

	template <typename T> ts::Tensor<T> parseTensor(
		std::ifstream in, ts::WengertList<T> * wList
	);

	template <typename T> std::string serializeTensorsVector(
		std::vector<ts::Tensor<T>> &tensorsVector
	);

	template <typename T> std::vector<ts::Tensor<T>> parseTensorsVector(
		std::ifstream in, ts::WengertList<T> * wList
	);
}
