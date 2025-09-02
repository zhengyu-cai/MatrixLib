// linear_algebra.h
#pragma once
#include "matrix_base.h"
#include <vector>

namespace matrixlib {

// 高斯消元法
template <typename T>
Matrix<T> gaussian_elimination(Matrix<T> matrix);

// LU 分解
template <typename T>
struct LUDecomposition {
    Matrix<T> L;
    Matrix<T> U;
    std::vector<size_t> pivot;
};

template <typename T>
LUDecomposition<T> lu_decomposition(const Matrix<T>& matrix);

// QR 分解
template <typename T>
struct QRDecomposition {
    Matrix<T> Q;
    Matrix<T> R;
};

template <typename T>
QRDecomposition<T> qr_decomposition(const Matrix<T>& matrix);

// 特征值和特征向量
template <typename T>
struct EigenResult {
    std::vector<T> eigenvalues;
    Matrix<T> eigenvectors;
};

template <typename T>
EigenResult<T> eigen(const Matrix<T>& matrix);

// 奇异值分解 (SVD)
template <typename T>
struct SVDResult {
    Matrix<T> U;
    Matrix<T> S;
    Matrix<T> V;
};

template <typename T>
SVDResult<T> svd(const Matrix<T>& matrix);

} // namespace matrixlib