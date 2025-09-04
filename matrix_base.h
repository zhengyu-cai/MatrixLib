// matrix_base.h
#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <initializer_list>
#include <cstddef>

namespace matrixlib {

template <typename T>
class Matrix {
protected:
    std::vector<T> data_;
    size_t rows_;
    size_t cols_;
    
public:
    using value_type = T;
    
    // 构造函数
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(rows * cols) {}
    Matrix(size_t rows, size_t cols, const T& init_value) : rows_(rows), cols_(cols), data_(rows * cols, init_value) {}
    Matrix(std::initializer_list<std::initializer_list<T>> init) {
        rows_ = init.size();
        cols_ = init.begin()->size();
        data_.reserve(rows_ * cols_);
        for (const auto& row : init) {
            for (const auto& elem : row) {
                data_.push_back(elem);
            }
        }
    }
    
    // 拷贝构造函数
    Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {}
    
    // 拷贝赋值运算符
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = other.data_;
        }
        return *this;
    }
    
    // 移动构造函数
    Matrix(Matrix&& other) noexcept : rows_(other.rows_), cols_(other.cols_), data_(std::move(other.data_)) {
        other.rows_ = 0;
        other.cols_ = 0;
    }
    
    // 移动赋值运算符
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = std::move(other.data_);
            other.rows_ = 0;
            other.cols_ = 0;
        }
        return *this;
    }
    
    // 基础访问方法
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }
    
    // 元素访问
    T& operator()(size_t row, size_t col) { return data_[row * cols_ + col]; }
    const T& operator()(size_t row, size_t col) const { return data_[row * cols_ + col]; }
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    
    // 迭代器支持
    typename std::vector<T>::iterator begin() { return data_.begin(); }
    typename std::vector<T>::const_iterator begin() const { return data_.begin(); }
    typename std::vector<T>::iterator end() { return data_.end(); }
    typename std::vector<T>::const_iterator end() const { return data_.end(); }
    
    virtual ~Matrix() = default;
};

} // namespace matrixlib  // 确保这里有这个关闭的大括号！