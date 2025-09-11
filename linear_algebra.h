#ifndef MATRIXLIB_MAIN_LINEAR_ALGEBRA_0910_H
#define MATRIXLIB_MAIN_LINEAR_ALGEBRA_0910_H

#include "matrix_base.h"
#include "matrix_operations.h"
#include "numerical_optimization.h"
#include "expression_templates.h"
#include "matrix_decomposition.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

namespace matrixlib
{

    template <typename T>
    Matrix<T> gaussian_elimination(Matrix<T> A, Matrix<T> b)
    {
        if (A.rows() != A.cols())
        {
            throw std::invalid_argument("矩阵A必须是方阵");
        }
        if (b.cols() != 1 || A.rows() != b.rows())
        {
            throw std::invalid_argument("向量b必须是列向量且与A行数相同");
        }

        size_t n = A.rows();

        Matrix<T> augmented(n, n + 1);
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                augmented(i, j) = A(i, j);
            }
            augmented(i, n) = b(i, 0);
        }

        for (size_t i = 0; i < n; i++)
        {
            size_t max_row = i;
            T max_val = std::abs(augmented(i, i));
            for (size_t k = i + 1; k < n; k++)
            {
                if (std::abs(augmented(k, i)) > max_val)
                {
                    max_val = std::abs(augmented(k, i));
                    max_row = k;
                }
            }

            if (max_row != i)
            {
                for (size_t j = i; j <= n; j++)
                {
                    std::swap(augmented(i, j), augmented(max_row, j));
                }
            }

            for (size_t k = i + 1; k < n; k++)
            {
                T factor = augmented(k, i) / augmented(i, i);
                for (size_t j = i; j <= n; j++)
                {
                    augmented(k, j) -= factor * augmented(i, j);
                }
            }
        }

        Matrix<T> x(n, 1);
        for (int i = n - 1; i >= 0; i--)
        {
            x(i, 0) = augmented(i, n);
            for (size_t j = i + 1; j < n; j++)
            {
                x(i, 0) -= augmented(i, j) * x(j, 0);
            }
            x(i, 0) /= augmented(i, i);
        }

        return x;
    }

    template <typename T>
    std::pair<Matrix<T>, Matrix<T>> lu_decomposition(const Matrix<T> &A)
    {
        if (A.rows() != A.cols())
        {
            throw std::invalid_argument("矩阵必须是方阵");
        }

        size_t n = A.rows();
        Matrix<T> L(n, n, 0);
        Matrix<T> U(n, n, 0);

        for (size_t i = 0; i < n; i++)
        {
            for (size_t k = i; k < n; k++)
            {
                T sum = 0;
                for (size_t j = 0; j < i; j++)
                {
                    sum += L(i, j) * U(j, k);
                }
                U(i, k) = A(i, k) - sum;
            }

            for (size_t k = i; k < n; k++)
            {
                if (i == k)
                {
                    L(i, i) = 1;
                }
                else
                {
                    T sum = 0;
                    for (size_t j = 0; j < i; j++)
                    {
                        sum += L(k, j) * U(j, i);
                    }
                    L(k, i) = (A(k, i) - sum) / U(i, i);
                }
            }
        }

        return {L, U};
    }

    template <typename T>
    std::pair<Matrix<T>, Matrix<T>> qr_decomposition(const Matrix<T> &A)
    {
        size_t m = A.rows();
        size_t n = A.cols();

        Matrix<T> Q(m, n, 0);
        Matrix<T> R(n, n, 0);

        for (size_t j = 0; j < n; j++)
        {
            Matrix<T> v(m, 1);
            for (size_t i = 0; i < m; i++)
            {
                v(i, 0) = A(i, j);
            }

            for (size_t k = 0; k < j; k++)
            {
                Matrix<T> q_k(m, 1);
                for (size_t i = 0; i < m; i++)
                {
                    q_k(i, 0) = Q(i, k);
                }

                R(k, j) = dot_product(q_k, v);
                for (size_t i = 0; i < m; i++)
                {
                    v(i, 0) -= R(k, j) * q_k(i, 0);
                }
            }

            T norm_v = norm(v);
            if (norm_v < std::numeric_limits<T>::epsilon())
            {
                throw std::runtime_error("矩阵秩不足");
            }

            R(j, j) = norm_v;
            for (size_t i = 0; i < m; i++)
            {
                Q(i, j) = v(i, 0) / norm_v;
            }
        }

        return {Q, R};
    }

    // 特征值和特征向量
    template <typename T>
    struct EigenResult {
    std::vector<T> eigenvalues;
    Matrix<T> eigenvectors;
    };

    
    template <typename T>
    std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> svd(const Matrix<T> &A)
    {
        size_t m = A.rows();
        size_t n = A.cols();
        const T eps = 1e-12;

        Matrix<T> A_transpose = transpose(A);
        Matrix<T> ATA = A_transpose * A;
        EigenResult<T> eigen_result = symmetric_eigen(ATA);
        std::vector<T> eigenvalues = eigen_result.eigenvalues;
        Matrix<T> V = eigen_result.eigenvectors;

        std::vector<size_t> indices(n);
        for (size_t i = 0; i < n; ++i)
            indices[i] = i;
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b)
                  { return eigenvalues[a] > eigenvalues[b]; });

        std::vector<T> sorted_eigenvalues(n);
        Matrix<T> sorted_V(n, n);
        for (size_t i = 0; i < n; ++i)
        {
            sorted_eigenvalues[i] = eigenvalues[indices[i]];
            // 复制排序后的特征向量列
            for (size_t j = 0; j < n; ++j)
            {
                sorted_V(j, i) = V(j, indices[i]);
            }
        }

        std::vector<T> singular_values(n);
        for (size_t i = 0; i < n; ++i)
        {
            if (sorted_eigenvalues[i] < 0)
            {
                // A^T A 是半正定矩阵，特征值理论上非负，此处处理数值误差
                singular_values[i] = 0;
            }
            else
            {
                singular_values[i] = std::sqrt(sorted_eigenvalues[i]);
            }
        }

        Matrix<T> Sigma(m, n, 0);
        for (size_t i = 0; i < std::min(m, n); ++i)
        {
            Sigma(i, i) = singular_values[i];
        }

        Matrix<T> U(m, m, 0);
        for (size_t i = 0; i < n; ++i)
        {
            if (singular_values[i] < 1e-12)
            {
                // 奇异值接近零，跳过（或用随机正交向量填充，此处简化处理）
                continue;
            }
            // 计算 U 的第i列：(A * V的第i列) / 奇异值
            Matrix<T> v_col(n, 1);
            for (size_t j = 0; j < n; ++j)
            {
                v_col(j, 0) = sorted_V(j, i);
            }
            Matrix<T> u_col = A * v_col * (1.0 / singular_values[i]);
            // 赋值给U的第i列
            for (size_t j = 0; j < m; ++j)
            {
                U(j, i) = u_col(j, 0);
            }
        }

        return {U, Sigma, sorted_V};
    }

}

#endif