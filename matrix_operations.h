// matrix_operations.h
#pragma once
#include "matrix_base.h"

namespace matrixlib {

// 矩阵加法
template <typename T>
Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs);

// 矩阵减法
template <typename T>
Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs);

// 标量乘法
template <typename T>
Matrix<T> operator*(const Matrix<T>& matrix, T scalar);
template <typename T>
Matrix<T> operator*(T scalar, const Matrix<T>& matrix);

// 矩阵乘法
template <typename T>
Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs);

// 逐元素乘法
template <typename T>
Matrix<T> elementwise_multiply(const Matrix<T>& lhs, const Matrix<T>& rhs);

// 矩阵转置
template <typename T>
Matrix<T> transpose(const Matrix<T>& matrix);

// 矩阵求逆（方阵）
template <typename T>
Matrix<T> inverse(const Matrix<T>& matrix);

// 矩阵行列式
template <typename T>
T determinant(const Matrix<T>& matrix);

} // namespace matrixlib