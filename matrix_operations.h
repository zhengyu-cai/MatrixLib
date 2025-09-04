#pragma once
#include "matrix_base.h"
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace matrixlib {

// 矩阵加法
template <typename T>
Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    Matrix<T> result(lhs.rows(), lhs.cols());
    for (size_t i = 0; i < lhs.rows(); i++) {
        for (size_t j = 0; j < lhs.cols(); j++) {
            result(i, j) = lhs(i, j) + rhs(i, j);
        }
    }
    return result;
}

// 矩阵减法
template <typename T>
Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    
    Matrix<T> result(lhs.rows(), lhs.cols());
    for (size_t i = 0; i < lhs.rows(); i++) {
        for (size_t j = 0; j < lhs.cols(); j++) {
            result(i, j) = lhs(i, j) - rhs(i, j);
        }
    }
    return result;
}

// 标量乘法
template <typename T>
Matrix<T> operator*(const Matrix<T>& matrix, T scalar) {
    Matrix<T> result(matrix.rows(), matrix.cols());
    for (size_t i = 0; i < matrix.rows(); i++) {
        for (size_t j = 0; j < matrix.cols(); j++) {
            result(i, j) = matrix(i, j) * scalar;
        }
    }
    return result;
}

template <typename T>
Matrix<T> operator*(T scalar, const Matrix<T>& matrix) {
    return matrix * scalar;
}

// 矩阵乘法
template <typename T>
Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    if (lhs.cols() != rhs.rows()) {
        throw std::invalid_argument("Matrix dimensions must match for multiplication: lhs.cols() == rhs.rows()");
    }
    
    Matrix<T> result(lhs.rows(), rhs.cols());
    for (size_t i = 0; i < lhs.rows(); i++) {
        for (size_t j = 0; j < rhs.cols(); j++) {
            T sum = T();
            for (size_t k = 0; k < lhs.cols(); k++) {
                sum += lhs(i, k) * rhs(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

// 逐元素乘法
template <typename T>
Matrix<T> elementwise_multiply(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication");
    }
    
    Matrix<T> result(lhs.rows(), lhs.cols());
    for (size_t i = 0; i < lhs.rows(); i++) {
        for (size_t j = 0; j < lhs.cols(); j++) {
            result(i, j) = lhs(i, j) * rhs(i, j);
        }
    }
    return result;
}

// 转置
template <typename T>
Matrix<T> transpose(const Matrix<T>& matrix) {
    Matrix<T> result(matrix.cols(), matrix.rows());
    for (size_t i = 0; i < matrix.rows(); i++) {
        for (size_t j = 0; j < matrix.cols(); j++) {
            result(j, i) = matrix(i, j);
        }
    }
    return result;
}

// 计算行列式的辅助函数（递归实现）
template <typename T>
T determinant_helper(const Matrix<T>& matrix) {
    const size_t n = matrix.rows();
    
    if (n == 1) {
        return matrix(0, 0);
    }
    if (n == 2) {
        return matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0);
    }
    
    T det = T();
    for (size_t col = 0; col < n; col++) {
        // 创建子矩阵（去掉第0行和第col列）
        Matrix<T> submatrix(n - 1, n - 1);
        for (size_t i = 1; i < n; i++) {
            size_t subcol = 0;
            for (size_t j = 0; j < n; j++) {
                if (j == col) continue;
                submatrix(i - 1, subcol++) = matrix(i, j);
            }
        }
        
        // 递归计算并累加
        T subdet = determinant_helper(submatrix);
        if (col % 2 == 0) {
            det += matrix(0, col) * subdet;
        } else {
            det -= matrix(0, col) * subdet;
        }
    }
    return det;
}

// 计算行列式
template <typename T>
T determinant(const Matrix<T>& matrix) {
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("Matrix must be square to compute determinant");
    }
    return determinant_helper(matrix);
}

// 计算余子式矩阵
template <typename T>
Matrix<T> cofactor_matrix(const Matrix<T>& matrix) {
    const size_t n = matrix.rows();
    if (n != matrix.cols()) {
        throw std::invalid_argument("Matrix must be square to compute cofactor matrix");
    }
    
    Matrix<T> cofactor(n, n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            // 创建子矩阵（去掉第i行和第j列）
            Matrix<T> submatrix(n - 1, n - 1);
            size_t subrow = 0;
            for (size_t row = 0; row < n; row++) {
                if (row == i) continue;
                size_t subcol = 0;
                for (size_t col = 0; col < n; col++) {
                    if (col == j) continue;
                    submatrix(subrow, subcol++) = matrix(row, col);
                }
                subrow++;
            }
            
            // 计算余子式
            T subdet = determinant_helper(submatrix);
            cofactor(i, j) = ((i + j) % 2 == 0 ? subdet : -subdet);
        }
    }
    return cofactor;
}

// 矩阵求逆
template <typename T>
Matrix<T> inverse(const Matrix<T>& matrix) {
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("Matrix must be square to compute inverse");
    }
    
    const size_t n = matrix.rows();
    T det = determinant(matrix);
    
    if (det == T()) {
        throw std::invalid_argument("Matrix is singular (determinant is zero), cannot compute inverse");
    }
    
    // 计算伴随矩阵（余子式矩阵的转置）
    Matrix<T> adjugate = transpose(cofactor_matrix(matrix));
    
    // 乘以行列式的倒数
    return adjugate * (T(1) / det);
}

// 矩阵输出（方便调试）
template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix) {
    os << "[";
    for (size_t i = 0; i < matrix.rows(); i++) {
        if (i > 0) os << " ";
        os << "[";
        for (size_t j = 0; j < matrix.cols(); j++) {
            os << matrix(i, j);
            if (j < matrix.cols() - 1) os << ", ";
        }
        os << "]";
        if (i < matrix.rows() - 1) os << "\n";
    }
    os << "]";
    return os;
}

}