/*
* Manual datatypes support for autodiff (the compiler would need an
* implementation directly in the header file otherwise).
* This method has the advantage of forbidding non-numeric data types like
* strings, etc... as well as keeping the source code in a dedicated .cpp file.
* More data types may be added in the future.
*/

#include "../include/autodiff.hpp"
#include "./autodiff.cpp"
#include "./autodiff_operations.cpp"


	// float

template class ts::Node<float>;
template class ts::WengertList<float>;
template class ts::Tensor<float>;
template class ts::Gradient<float>;

template ts::Tensor<float> ts::NewTensor(
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> newValue,
	ts::WengertList<float> * newWList
);

template ts::Tensor<float> ts::operator+(const ts::Tensor<float> &x, const ts::Tensor<float> &y);
template ts::Tensor<float> ts::operator-(const ts::Tensor<float> &x, const ts::Tensor<float> &y);
template ts::Tensor<float> ts::operator*(const ts::Tensor<float> &x, const ts::Tensor<float> &y);
template ts::Tensor<float> ts::operator/(const ts::Tensor<float> &x, const ts::Tensor<float> &y);

template ts::Tensor<float> ts::matProd(const ts::Tensor<float> &x, const ts::Tensor<float> &y);



	// double

template class ts::Node<double>;
template class ts::WengertList<double>;
template class ts::Tensor<double>;
template class ts::Gradient<double>;

template ts::Tensor<double> ts::NewTensor(
	Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> newValue,
	ts::WengertList<double> * newWList
);

template ts::Tensor<double> ts::operator+(const ts::Tensor<double> &x, const ts::Tensor<double> &y);
template ts::Tensor<double> ts::operator-(const ts::Tensor<double> &x, const ts::Tensor<double> &y);
template ts::Tensor<double> ts::operator*(const ts::Tensor<double> &x, const ts::Tensor<double> &y);
template ts::Tensor<double> ts::operator/(const ts::Tensor<double> &x, const ts::Tensor<double> &y);

template ts::Tensor<double> ts::matProd(const ts::Tensor<double> &x, const ts::Tensor<double> &y);
