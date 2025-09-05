#pragma once
#include "matrix_base.h"
#include <cmath>
#include <stdexcept>
#include "matrix_operations.h"

namespace matrixlib
{
    // 向量点积
    template <typename T>
    T dot_product(const Matrix<T> &a, const Matrix<T> &b)
    {
        if (a.rows() != b.rows() || a.cols() != 1 || b.cols() != 1)
        {
            throw std::invalid_argument("Matrix sizes do not match for dot product");
        }

        T result = 0;
        for (size_t i = 0; i < a.rows(); ++i)
        {
            result += a(i, 0) * b(i, 0);
        }
        
        return result;
    }

    // 计算向量的L2范数
    template <typename T>
    T norm(const Matrix<T> &a)
    {
        if (a.cols() != 1)
        {
            std::invalid_argument("Vector a must be a column vector");
        }
        return std::sqrt(dot_product(a, a));
    }

    // 共轭梯度法
    template <typename T>
    Matrix<T> conjugate_gradient(const Matrix<T> &A, const Matrix<T> &b,
                                 size_t max_iterations = 1000, T tolerance = 1e-10)
    {
        if (A.rows() != A.cols())
        {
            throw std::invalid_argument("Matrix A must be square");
        }

        if (b.cols() != 1 || A.rows() != b.rows())
        {
            throw std::invalid_argument("Vector b must be a column vector with the same number of rows as A");
        }

        size_t n = A.rows();
        Matrix<T> x(n, 1, 0); // 初始化解向量
        Matrix<T> r = b;
        Matrix<T> p = r;
        T r_dot = dot_product(r, r);

        for (size_t i = 0; i < max_iterations; ++i)
        {
            if (std::sqrt(r_dot) < tolerance)
            {
                break;
            }

            Matrix<T> Ap = A * p;
            T alpha = r_dot / dot_product(p, Ap);
            x = x + alpha * p;

            Matrix<T> r_new = r - alpha * Ap;
            T r_new_dot = dot_product(r_new, r_new);

            T beta = r_new_dot / r_dot;
            p = r_new + beta * p;

            r = r_new;
            r_dot = r_new_dot;
        }

        return x;
    }

    // 雅可比迭代
    template <typename T>
    Matrix<T> jacobi_iteration(const Matrix<T> &A, const Matrix<T> &b,
                               size_t max_iterations = 1000, T tolerance = 1e-10)
    {
        if (A.cols() != A.rows())
        {
            std::invalid_argument("Matrix A must be square");
        }

        if (b.cols() != 1 || A.cols() != b.rows())
        {
            std::invalid_argument("Vector b must be a column vector with the same number of rows as A");
        }

        for (size_t i = 0; i < A.cols(); ++i)
        {
            if (std::abs(A(i, i)) < 1e-15)
            {
                std::invalid_argument("Matrix A has zero diagonal element, Jacobi iteration not applicable");
            }
        }

        size_t n = A.cols();
        Matrix<T> x(n, 1, 0);
        Matrix<T> x_new(n, 1);

        for (size_t i = 0; i < max_iterations; ++i)
        {
            // 迭代计算新解
            for (size_t j = 0; j < n; ++j)
            {
                T sum = 0;
                for (size_t k = 0; k < n; ++k)
                {
                    if (j != k)
                    {
                        sum += A(j, k) * x(k, 0);
                    }
                }
                x_new(j, 0) = (b(j, 0) - sum) / A(j, j);
            }

            // 计算误差，检查收敛条件
            Matrix<T> e = x_new - x;
            if (norm(e) < tolerance)
            {
                break;
            }

            x = x_new;
        }

        return x;
    }

    // 高斯-赛德尔迭代
    template <typename T>
    Matrix<T> gauss_seidel_iteration(const Matrix<T> &A, const Matrix<T> &b,
                                     size_t max_iterations = 1000, T tolerance = 1e-10)
    {
        if (A.cols() != A.rows())
        {
            std::invalid_argument("Matrix A must be square");
        }

        if (b.cols() != 1 || A.cols() != b.rows())
        {
            std::invalid_argument("Vector b must be a column vector with the same number of rows as A");
        }

        for (size_t i = 0; i < A.cols(); ++i)
        {
            if (std::abs(A(i, i)) < 1e-15)
            {
                std::invalid_argument("Matrix A has zero diagonal element, Jacobi iteration not applicable");
            }
        }

        size_t n = A.cols();
        Matrix<T> x(n, 1, 0);
        Matrix<T> x_old(n, 1);

        for (size_t iter = 0; iter < max_iterations; ++iter)
        {
            x_old = x;
            // 迭代计算新解
            for (size_t i = 0; i < n; ++i)
            {
                T sum1 = 0, sum2 = 0;

                for (size_t j = 0; j < i; ++j)
                {
                    sum1 += A(i, j) * x(j, 0);
                }

                for (size_t j = i + 1; j < n; ++j)
                {
                    sum2 += A(i, j) * x_old(j, 0);
                }

                x(i, 0) = (b(i, 0) - sum1 - sum2) / A(i, i);
            }

            Matrix<T> e = x - x_old;
            if (norm(e) < tolerance)
            {
                break;
            }
        }

        return x;
    }

    // 最小二乘法
    template <typename T>
    Matrix<T> least_squares(const Matrix<T> &A, const Matrix<T> &b)
    {
        if (b.cols() != 1 || A.rows() != b.rows())
        {
            std::invalid_argument("Vector b must be a column vector with the same number of rows as A");
        }

        // 计算AtA和ATb
        Matrix<T> A_transpose = transpose(A);
        Matrix<T> AtA = A_transpose * A;
        Matrix<T> ATb = A_transpose * b;

        // 使用共轭梯度法求解
        return conjugate_gradient(AtA, ATb);
    }

} // namespace matrixlib