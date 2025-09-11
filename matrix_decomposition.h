#pragma once
#include "matrix_base.h"
#include "linear_algebra.h"
#include <random>
#include <cmath>
#include <stdexcept>

namespace matrixlib
{

    // 函数前置声明
    template <typename T>
    Matrix<T> identity_matrix(size_t n);

    // 乔列斯基分解
    template <typename T>
    Matrix<T> cholesky_decomposition(const Matrix<T> &matrix)
    {
        if (matrix.rows() != matrix.cols())
        {
            throw std::invalid_argument("Matrix must be square for Cholesky decomposition");
        }

        size_t n = matrix.rows();
        Matrix<T> L(n, n);

        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j <= i; j++)
            {
                T sum = 0;
                for (size_t k = 0; k < j; k++)
                {
                    sum += L(i, k) * L(j, k);
                }

                if (i == j)
                {
                    T diag = matrix(i, i) - sum;
                    if (diag <= 0)
                    {
                        throw std::invalid_argument("Matrix is not positive definite");
                    }
                    L(i, j) = std::sqrt(diag);
                }
                else
                {
                    L(i, j) = (matrix(i, j) - sum) / L(j, j);
                }
            }
        }

        return L;
    }

    // 对称矩阵特征值分解（使用Jacobi方法）
    template <typename T>
    EigenResult<T> symmetric_eigen(const Matrix<T> &matrix)
    {
        if (matrix.rows() != matrix.cols())
        {
            throw std::invalid_argument("Matrix must be square for eigenvalue decomposition");
        }

        size_t n = matrix.rows();
        Matrix<T> A = matrix;
        Matrix<T> eigenvectors = identity_matrix<T>(n);
        const size_t max_iterations = 100;
        const T tolerance = 1e-10;

        for (size_t iter = 0; iter < max_iterations; iter++)
        {
            T max_off_diag = 0;
            size_t p = 0, q = 0;

            // 寻找最大的非对角元素
            for (size_t i = 0; i < n; i++)
            {
                for (size_t j = i + 1; j < n; j++)
                {
                    if (std::abs(A(i, j)) > max_off_diag)
                    {
                        max_off_diag = std::abs(A(i, j));
                        p = i;
                        q = j;
                    }
                }
            }

            if (max_off_diag < tolerance)
            {
                break;
            }

            // 计算旋转角度
            T theta = (A(q, q) - A(p, p)) / (2 * A(p, q));
            T t = std::abs(theta) < std::numeric_limits<T>::max() ? (theta >= 0 ? 1.0 / (theta + std::sqrt(1 + theta * theta)) : 1.0 / (theta - std::sqrt(1 + theta * theta))) : 0.5 / theta;

            T c = 1.0 / std::sqrt(1 + t * t);
            T s = t * c;

            // 应用旋转
            for (size_t r = 0; r < n; r++)
            {
                T temp = A(r, p);
                A(r, p) = c * temp - s * A(r, q);
                A(r, q) = s * temp + c * A(r, q);
            }

            for (size_t r = 0; r < n; r++)
            {
                T temp = A(p, r);
                A(p, r) = c * temp - s * A(q, r);
                A(q, r) = s * temp + c * A(q, r);

                temp = eigenvectors(r, p);
                eigenvectors(r, p) = c * temp - s * eigenvectors(r, q);
                eigenvectors(r, q) = s * temp + c * eigenvectors(r, q);
            }
        }

        // 提取特征值
        std::vector<T> eigenvalues(n);
        for (size_t i = 0; i < n; i++)
        {
            eigenvalues[i] = A(i, i);
        }

        return {eigenvalues, eigenvectors};
    }

    // 生成单位矩阵
    template <typename T>
    Matrix<T> identity_matrix(size_t n)
    {
        Matrix<T> result(n, n);
        for (size_t i = 0; i < n; i++)
        {
            result(i, i) = 1;
        }
        return result;
    }

    // 生成对角矩阵
    template <typename T>
    Matrix<T> diagonal_matrix(const std::vector<T> &diagonal)
    {
        size_t n = diagonal.size();
        Matrix<T> result(n, n);
        for (size_t i = 0; i < n; i++)
        {
            result(i, i) = diagonal[i];
        }
        return result;
    }

    // 生成随机矩阵
    template <typename T>
    Matrix<T> random_matrix(size_t rows, size_t cols, T min, T max)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(min, max);

        Matrix<T> result(rows, cols);
        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < cols; j++)
            {
                result(i, j) = dis(gen);
            }
        }
        return result;
    }

    // 带状矩阵
    template <typename T>
    class BandMatrix : public Matrix<T>
    {
    private:
        size_t lower_band_;
        size_t upper_band_;
        std::vector<T> data_; // 压缩存储的数据

        size_t get_index(size_t i, size_t j) const
        {
            if (j > i + lower_band_ || i > j + upper_band_)
            {
                throw std::out_of_range("Element outside band");
            }
            return (i - j + lower_band_) * this->cols() + j;
        }

    public:
        BandMatrix(size_t size, size_t lower_band, size_t upper_band)
            : Matrix<T>(size, size), lower_band_(lower_band), upper_band_(upper_band),
              data_((lower_band + upper_band + 1) * size) {}

        T &operator()(size_t row, size_t col)
        {
            if (col > row + lower_band_ || row > col + upper_band_)
            {
                throw std::out_of_range("Element outside band");
            }
            return data_[get_index(row, col)];
        }

        const T &operator()(size_t row, size_t col) const
        {
            if (col > row + lower_band_ || row > col + upper_band_)
            {
                throw std::out_of_range("Element outside band");
            }
            return data_[get_index(row, col)];
        }

        // 特殊化的矩阵向量乘法
        std::vector<T> multiply(const std::vector<T> &vec) const
        {
            if (vec.size() != this->cols())
            {
                throw std::invalid_argument("Vector size must match matrix dimensions");
            }

            std::vector<T> result(this->rows(), 0);
            for (size_t i = 0; i < this->rows(); i++)
            {
                size_t j_start = i > lower_band_ ? i - lower_band_ : 0;
                size_t j_end = std::min(i + upper_band_ + 1, this->cols());

                for (size_t j = j_start; j < j_end; j++)
                {
                    result[i] += (*this)(i, j) * vec[j];
                }
            }
            return result;
        }
    };

}