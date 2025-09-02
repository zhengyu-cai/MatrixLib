// numerical_optimization.h
#pragma once
#include "matrix_base.h"

namespace matrixlib {

// 共轭梯度法
template <typename T>
Matrix<T> conjugate_gradient(const Matrix<T>& A, const Matrix<T>& b, 
                           size_t max_iterations = 1000, T tolerance = 1e-10);

// 雅可比迭代
template <typename T>
Matrix<T> jacobi_iteration(const Matrix<T>& A, const Matrix<T>& b, 
                         size_t max_iterations = 1000, T tolerance = 1e-10);

// 高斯-赛德尔迭代
template <typename T>
Matrix<T> gauss_seidel_iteration(const Matrix<T>& A, const Matrix<T>& b, 
                               size_t max_iterations = 1000, T tolerance = 1e-10);

// 最小二乘法
template <typename T>
Matrix<T> least_squares(const Matrix<T>& A, const Matrix<T>& b);

} // namespace matrixlib

template <typename T>
matrixlib::Matrix<T> conjugate_gradient(const matrixlib::Matrix<T>& A, const matrixlib::Matrix<T>& b, 
                                      size_t max_iterations = 1000, T tolerance = 1e-10) {}