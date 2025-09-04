// expression_templates.h
#pragma once
#include <cstddef>
#include "matrix_base.h"

namespace matrixlib {

// 表达式模板基类
template <typename E, typename T>
struct MatrixExpression {
    using value_type = T;
    
    const E& derived() const { return static_cast<const E&>(*this); }
    size_t rows() const { return derived().rows(); }
    size_t cols() const { return derived().cols(); }
    T operator()(size_t i, size_t j) const { return derived()(i, j); }
};

// 矩阵包装器
template <typename T>
class MatrixWrapper : public MatrixExpression<MatrixWrapper<T>, T> {
private:
    const Matrix<T>& matrix_;
public:
    MatrixWrapper(const Matrix<T>& matrix) : matrix_(matrix) {}
    size_t rows() const { return matrix_.rows(); }
    size_t cols() const { return matrix_.cols(); }
    T operator()(size_t i, size_t j) const { return matrix_(i, j); }
};

// 加法表达式
template <typename E1, typename E2, typename T>
class MatrixAdd : public MatrixExpression<MatrixAdd<E1, E2, T>, T> {
private:
    const E1& lhs_;
    const E2& rhs_;
public:
    MatrixAdd(const E1& lhs, const E2& rhs) : lhs_(lhs), rhs_(rhs) {
        if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }
    }
    size_t rows() const { return lhs_.rows(); }
    size_t cols() const { return lhs_.cols(); }
    T operator()(size_t i, size_t j) const { return lhs_(i, j) + rhs_(i, j); }
};

// 减法表达式
template <typename E1, typename E2, typename T>
class MatrixSub : public MatrixExpression<MatrixSub<E1, E2, T>, T> {
private:
    const E1& lhs_;
    const E2& rhs_;
public:
    MatrixSub(const E1& lhs, const E2& rhs) : lhs_(lhs), rhs_(rhs) {
        if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }
    }
    size_t rows() const { return lhs_.rows(); }
    size_t cols() const { return lhs_.cols(); }
    T operator()(size_t i, size_t j) const { return lhs_(i, j) - rhs_(i, j); }
};

// 矩阵乘法表达式
template <typename E1, typename E2, typename T>
class MatrixMul : public MatrixExpression<MatrixMul<E1, E2, T>, T> {
private:
    const E1& lhs_;
    const E2& rhs_;
public:
    MatrixMul(const E1& lhs, const E2& rhs) : lhs_(lhs), rhs_(rhs) {
        if (lhs.cols() != rhs.rows()) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication");
        }
    }
    size_t rows() const { return lhs_.rows(); }
    size_t cols() const { return rhs_.cols(); }
    T operator()(size_t i, size_t j) const {
        T sum = T();
        for (size_t k = 0; k < lhs_.cols(); ++k) {
            sum += lhs_(i, k) * rhs_(k, j);
        }
        return sum;
    }
};

// 矩阵加法函数
template <typename T>
Matrix<T> matrix_add(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    MatrixWrapper<T> lhs_wrapper(lhs);
    MatrixWrapper<T> rhs_wrapper(rhs);
    MatrixAdd<MatrixWrapper<T>, MatrixWrapper<T>, T> expr(lhs_wrapper, rhs_wrapper);
    
    Matrix<T> result(lhs.rows(), lhs.cols());
    for (size_t i = 0; i < result.rows(); ++i) {
        for (size_t j = 0; j < result.cols(); ++j) {
            result(i, j) = expr(i, j);
        }
    }
    return result;
}

// 矩阵减法函数
template <typename T>
Matrix<T> matrix_subtract(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    MatrixWrapper<T> lhs_wrapper(lhs);
    MatrixWrapper<T> rhs_wrapper(rhs);
    MatrixSub<MatrixWrapper<T>, MatrixWrapper<T>, T> expr(lhs_wrapper, rhs_wrapper);
    
    Matrix<T> result(lhs.rows(), lhs.cols());
    for (size_t i = 0; i < result.rows(); ++i) {
        for (size_t j = 0; j < result.cols(); ++j) {
            result(i, j) = expr(i, j);
        }
    }
    return result;
}

// 矩阵乘法函数
template <typename T>
Matrix<T> matrix_multiply(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    MatrixWrapper<T> lhs_wrapper(lhs);
    MatrixWrapper<T> rhs_wrapper(rhs);
    MatrixMul<MatrixWrapper<T>, MatrixWrapper<T>, T> expr(lhs_wrapper, rhs_wrapper);
    
    Matrix<T> result(lhs.rows(), rhs.cols());
    for (size_t i = 0; i < result.rows(); ++i) {
        for (size_t j = 0; j < result.cols(); ++j) {
            result(i, j) = expr(i, j);
        }
    }
    return result;
}

} // namespace matrixlib