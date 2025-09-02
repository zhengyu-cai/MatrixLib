// expression_templates.h
#pragma once
#include <type_traits>

namespace matrixlib {

// 表达式模板基类
template <typename E>
struct MatrixExpression {
    const E& derived() const { return static_cast<const E&>(*this); }
    size_t rows() const { return derived().rows(); }
    size_t cols() const { return derived().cols(); }
    T operator()(size_t i, size_t j) const { return derived()(i, j); }
};

// 矩阵包装器
template <typename T>
class MatrixWrapper : public MatrixExpression<MatrixWrapper<T>> {
private:
    const Matrix<T>& matrix_;
public:
    MatrixWrapper(const Matrix<T>& matrix) : matrix_(matrix) {}
    size_t rows() const { return matrix_.rows(); }
    size_t cols() const { return matrix_.cols(); }
    T operator()(size_t i, size_t j) const { return matrix_(i, j); }
};

// 加法表达式
template <typename E1, typename E2>
class MatrixAdd : public MatrixExpression<MatrixAdd<E1, E2>> {
private:
    const E1& lhs_;
    const E2& rhs_;
public:
    MatrixAdd(const E1& lhs, const E2& rhs) : lhs_(lhs), rhs_(rhs) {}
    size_t rows() const { return lhs_.rows(); }
    size_t cols() const { return lhs_.cols(); }
    T operator()(size_t i, size_t j) const { return lhs_(i, j) + rhs_(i, j); }
};

// 运算符重载使用表达式模板
template <typename E1, typename E2>
MatrixAdd<E1, E2> operator+(const MatrixExpression<E1>& lhs, 
                           const MatrixExpression<E2>& rhs) {
    return MatrixAdd<E1, E2>(lhs.derived(), rhs.derived());
}

} // namespace matrixlib