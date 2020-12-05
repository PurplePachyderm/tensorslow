/*
* Set of helper functions to implement serialization and parsing of ts::Model
* derived classes.
*/

#pragma once

#include "utils.hpp"
#include "autodiff.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>


namespace ts {
	template <typename T> std::string serializeTensor(ts::Tensor<T> &tensor);

	template <typename T> ts::Tensor<T> parseTensor(
		std::ifstream &in, ts::WengertList<T> * wList
	);

	template <typename T> std::string serializeTensorsVector(
		std::vector<ts::Tensor<T>> &tensorsVector
	);

	template <typename T> std::vector<ts::Tensor<T>> parseTensorsVector(
		std::ifstream &in, ts::WengertList<T> * wList
	);
}
