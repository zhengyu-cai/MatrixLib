// matrix_base.h
#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <initializer_list>

namespace matrixlib {

template <typename T>
class Matrix {
protected:
    std::vector<T> data_;
    size_t rows_;
    size_t cols_;
    
public:
    // 构造函数
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, const T& init_value);
    Matrix(std::initializer_list<std::initializer_list<T>> init);
    
    // 拷贝构造函数和赋值运算符
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    
    // 移动构造函数和赋值运算符
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;
    
    // 基础访问方法
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }
    
    // 元素访问
    T& operator()(size_t row, size_t col);
    const T& operator()(size_t row, size_t col) const;
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    
    // 迭代器支持
    typename std::vector<T>::iterator begin() { return data_.begin(); }
    typename std::vector<T>::const_iterator begin() const { return data_.begin(); }
    typename std::vector<T>::iterator end() { return data_.end(); }
    typename std::vector<T>::const_iterator end() const { return data_.end(); }
    
    virtual ~Matrix() = default;
};

} // namespace matrixlib