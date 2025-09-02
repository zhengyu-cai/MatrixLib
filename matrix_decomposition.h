// matrix_decomposition.h
#pragma once
#include "matrix_base.h"

namespace matrixlib {

// 乔列斯基分解
template <typename T>
Matrix<T> cholesky_decomposition(const Matrix<T>& matrix);

// 对称矩阵特征值分解
template <typename T>
EigenResult<T> symmetric_eigen(const Matrix<T>& matrix);

// 生成特殊矩阵
template <typename T>
Matrix<T> identity_matrix(size_t n);

template <typename T>
Matrix<T> diagonal_matrix(const std::vector<T>& diagonal);

template <typename T>
Matrix<T> random_matrix(size_t rows, size_t cols, T min, T max);

// 带状矩阵
template <typename T>
class BandMatrix : public Matrix<T> {
private:
    size_t lower_band_;
    size_t upper_band_;
    
public:
    BandMatrix(size_t size, size_t lower_band, size_t upper_band);
    // 特殊化的存储和运算方法
};

} // namespace matrixlib