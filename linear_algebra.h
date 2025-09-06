#ifndef LINEAR_H
#define LINEAR_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <limits>

template <typename T>
class Matrix {
protected:
    std::vector<T> data_;
    size_t rows_;
    size_t cols_;

public:
    Matrix(size_t rows, size_t cols, const T& init_val = T())
        : rows_(rows), cols_(cols), data_(rows * cols, init_val) {}

    Matrix(size_t rows, size_t cols, const std::vector<T>& data)
        : rows_(rows), cols_(cols), data_(data) {
        if (data.size() != rows * cols) {
            throw std::invalid_argument("数据大小与矩阵维度不匹配");
        }
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    T& operator()(size_t row, size_t col) {
        return data_[row * cols_ + col];
    }

    const T& operator()(size_t row, size_t col) const {
        return data_[row * cols_ + col];
    }

    void print(const std::string& name = "") const {
        if (!name.empty()) {
            std::cout << name << " = " << std::endl;
        }

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                std::cout << std::setw(12) << std::setprecision(6)
                          << std::fixed << data_[i * cols_ + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::vector<T> operator*(const std::vector<T>& vec) const {
        if (cols_ != vec.size()) {
            throw std::invalid_argument("矩阵和向量维度不匹配");
        }

        std::vector<T> result(rows_, 0.0);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result[i] += (*this)(i, j) * vec[j];
            }
        }
        return result;
    }

    static T dotProduct(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("向量点积要求大小相同");
        }

        T result = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    static T vectorNorm(const std::vector<T>& vec) {
        T norm = 0.0;
        for (const auto& val : vec) {
            norm += val * val;
        }
        return std::sqrt(norm);
    }

    static std::vector<T> normalize(const std::vector<T>& vec) {
        T norm = vectorNorm(vec);
        if (norm < 1e-10) {
            throw std::runtime_error("无法对零向量归一化");
        }

        std::vector<T> result(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            result[i] = vec[i] / norm;
        }
        return result;
    }

    std::vector<T> getColumn(size_t col) const {
        std::vector<T> column(rows_);
        for (size_t i = 0; i < rows_; ++i) {
            column[i] = (*this)(i, col);
        }
        return column;
    }

    void setColumn(size_t col, const std::vector<T>& column) {
        if (column.size() != rows_) {
            throw std::invalid_argument("列向量大小必须与矩阵行数匹配");
        }
        for (size_t i = 0; i < rows_; ++i) {
            (*this)(i, col) = column[i];
        }
    }

    std::vector<T> getRow(size_t row) const {
        std::vector<T> row_vec(cols_);
        for (size_t j = 0; j < cols_; ++j) {
            row_vec[j] = (*this)(row, j);
        }
        return row_vec;
    }

    void setRow(size_t row, const std::vector<T>& row_vec) {
        if (row_vec.size() != cols_) {
            throw std::invalid_argument("行向量大小必须与矩阵列数匹配");
        }
        for (size_t j = 0; j < cols_; ++j) {
            (*this)(row, j) = row_vec[j];
        }
    }

    Matrix<T> transpose() const {
        Matrix<T> result(cols_, rows_, 0.0);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    static Matrix<T> multiply(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.cols() != B.rows()) {
            throw std::invalid_argument("矩阵维度不匹配，无法相乘");
        }

        Matrix<T> result(A.rows(), B.cols(), 0.0);
        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < B.cols(); ++j) {
                for (size_t k = 0; k < A.cols(); ++k) {
                    result(i, j) += A(i, k) * B(k, j);
                }
            }
        }
        return result;
    }

    static std::vector<T> scalarMultiply(const std::vector<T>& v, T scalar) {
        std::vector<T> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = v[i] * scalar;
        }
        return result;
    }

    static std::vector<T> vectorAdd(const std::vector<T>& v1, const std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            throw std::invalid_argument("向量加法要求大小相同");
        }
        std::vector<T> result(v1.size());
        for (size_t i = 0; i < v1.size(); ++i) {
            result[i] = v1[i] + v2[i];
        }
        return result;
    }

    static std::vector<T> vectorSubtract(const std::vector<T>& v1, const std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            throw std::invalid_argument("向量减法要求大小相同");
        }
        std::vector<T> result(v1.size());
        for (size_t i = 0; i < v1.size(); ++i) {
            result[i] = v1[i] - v2[i];
        }
        return result;
    }

    T frobeniusNorm() const {
        T sum = 0;
        for (const auto& elem : data_) {
            sum += elem * elem;
        }
        return std::sqrt(sum);
    }

    bool isSymmetric() const {
        if (rows_ != cols_) return false;

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = i + 1; j < cols_; ++j) {
                if (std::abs((*this)(i, j) - (*this)(j, i)) > 1e-10) {
                    return false;
                }
            }
        }
        return true;
    }

    static std::vector<T> gaussElimination(const Matrix<T>& A, const std::vector<T>& b) {
        if (A.rows() != A.cols()) {
            throw std::invalid_argument("高斯消元法要求矩阵A为方阵");
        }
        if (A.rows() != b.size()) {
            throw std::invalid_argument("矩阵A和向量b维度不兼容");
        }

        size_t n = A.rows();

        Matrix<T> augmented(n, n + 1);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                augmented(i, j) = A(i, j);
            }
            augmented(i, n) = b[i];
        }

        std::cout << "增广矩阵:" << std::endl;
        augmented.print();

        for (size_t pivot = 0; pivot < n; ++pivot) {
            size_t max_row = pivot;
            T max_val = std::abs(augmented(pivot, pivot));

            for (size_t i = pivot + 1; i < n; ++i) {
                T current_val = std::abs(augmented(i, pivot));
                if (current_val > max_val) {
                    max_val = current_val;
                    max_row = i;
                }
            }

            if (max_row != pivot) {
                std::cout << "交换行 " << pivot << " 和行 " << max_row << std::endl;
                for (size_t j = pivot; j <= n; ++j) {
                    std::swap(augmented(pivot, j), augmented(max_row, j));
                }
                augmented.print("交换后");
            }

            if (std::abs(augmented(pivot, pivot)) < 1e-10) {
                throw std::runtime_error("矩阵奇异或接近奇异");
            }

            for (size_t i = pivot + 1; i < n; ++i) {
                T factor = augmented(i, pivot) / augmented(pivot, pivot);
                for (size_t j = pivot; j <= n; ++j) {
                    augmented(i, j) -= factor * augmented(pivot, j);
                }
            }

            std::cout << "消元步骤 " << pivot + 1 << " 后:" << std::endl;
            augmented.print();
        }

        std::vector<T> x(n);
        for (int i = n - 1; i >= 0; --i) {
            x[i] = augmented(i, n);
            for (size_t j = i + 1; j < n; ++j) {
                x[i] -= augmented(i, j) * x[j];
            }
            x[i] /= augmented(i, i);
        }

        return x;
    }

    static std::pair<Matrix<T>, Matrix<T>> luDecomposition(const Matrix<T>& A) {
        if (A.rows() != A.cols()) {
            throw std::invalid_argument("LU分解要求矩阵为方阵");
        }

        size_t n = A.rows();
        Matrix<T> L(n, n, 0.0);
        Matrix<T> U(n, n, 0.0);

        std::cout << "开始矩阵的LU分解:" << std::endl;
        A.print("A");

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                U(i, j) = A(i, j);
                for (size_t k = 0; k < i; ++k) {
                    U(i, j) -= L(i, k) * U(k, j);
                }
            }

            for (size_t j = i; j < n; ++j) {
                if (i == j) {
                    L(i, i) = 1.0;
                } else {
                    L(j, i) = A(j, i);
                    for (size_t k = 0; k < i; ++k) {
                        L(j, i) -= L(j, k) * U(k, i);
                    }
                    if (std::abs(U(i, i)) < 1e-10) {
                        throw std::runtime_error("矩阵奇异，LU分解失败");
                    }
                    L(j, i) /= U(i, i);
                }
            }

            std::cout << "步骤 " << i + 1 << " 后:" << std::endl;
            L.print("L");
            U.print("U");
        }

        return {L, U};
    }

    static std::vector<T> solveLU(const Matrix<T>& A, const std::vector<T>& b) {
        if (A.rows() != A.cols()) {
            throw std::invalid_argument("矩阵A必须为方阵");
        }
        if (A.rows() != b.size()) {
            throw std::invalid_argument("矩阵A和向量b维度不兼容");
        }

        size_t n = A.rows();

        auto [L, U] = luDecomposition(A);

        std::cout << "LU分解完成:" << std::endl;
        L.print("L 矩阵");
        U.print("U 矩阵");

        Matrix<T> reconstructed(n, n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                for (size_t k = 0; k < n; ++k) {
                    reconstructed(i, j) += L(i, k) * U(k, j);
                }
            }
        }
        reconstructed.print("L * U (应等于 A)");

        std::vector<T> y(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            y[i] = b[i];
            for (size_t j = 0; j < i; ++j) {
                y[i] -= L(i, j) * y[j];
            }
            y[i] /= L(i, i);
        }

        std::cout << "中间解 y:" << std::endl;
        for (size_t i = 0; i < n; ++i) {
            std::cout << "y[" << i << "] = " << y[i] << std::endl;
        }

        std::vector<T> x(n, 0.0);
        for (int i = n - 1; i >= 0; --i) {
            x[i] = y[i];
            for (size_t j = i + 1; j < n; ++j) {
                x[i] -= U(i, j) * x[j];
            }
            x[i] /= U(i, i);
        }

        return x;
    }

    static std::pair<Matrix<T>, Matrix<T>> qrDecomposition(const Matrix<T>& A) {
        if (A.rows() < A.cols()) {
            throw std::invalid_argument("QR分解要求矩阵行数 >= 列数");
        }

        size_t m = A.rows();
        size_t n = A.cols();

        Matrix<T> Q(m, n, 0.0);
        Matrix<T> R(n, n, 0.0);

        std::cout << "开始使用Gram-Schmidt过程进行QR分解:" << std::endl;
        A.print("输入矩阵 A");

        for (size_t j = 0; j < n; ++j) {
            std::cout << "\n处理第 " << j + 1 << " 列:" << std::endl;

            std::vector<T> a_j = A.getColumn(j);
            std::vector<T> q_j = a_j;

            for (size_t i = 0; i < j; ++i) {
                std::vector<T> q_i = Q.getColumn(i);
                T r_ij = dotProduct(q_i, a_j);
                R(i, j) = r_ij;

                std::vector<T> projection = scalarMultiply(q_i, r_ij);
                q_j = vectorSubtract(q_j, projection);

                std::cout << "减去在 q" << i + 1
                          << " 上的投影 (r_" << i + 1 << j + 1 << " = " << r_ij << ")" << std::endl;
            }

            T norm = vectorNorm(q_j);
            if (norm < 1e-10) {
                throw std::runtime_error("矩阵秩亏，QR分解失败");
            }

            R(j, j) = norm;
            q_j = scalarMultiply(q_j, 1.0 / norm);
            Q.setColumn(j, q_j);

            std::cout << "剩余向量的范数: " << norm << std::endl;
            std::cout << "正交向量 q" << j + 1 << " 计算完成" << std::endl;

            Q.print("当前 Q 矩阵");
            R.print("当前 R 矩阵");
        }

        return {Q, R};
    }

    static std::vector<T> solveLeastSquares(const Matrix<T>& A, const std::vector<T>& b) {
        if (A.rows() != b.size()) {
            throw std::invalid_argument("矩阵A和向量b维度不兼容");
        }

        std::cout << "使用QR分解求解最小二乘问题:" << std::endl;
        std::cout << "min ||Ax - b||?" << std::endl;

        auto [Q, R] = qrDecomposition(A);

        std::cout << "QR分解完成:" << std::endl;
        Q.print("正交矩阵 Q");
        R.print("上三角矩阵 R");

        Matrix<T> reconstructed = multiply(Q, R);
        reconstructed.print("Q * R (应等于 A)");

        std::vector<T> qt_b(R.cols(), 0.0);
        for (size_t j = 0; j < R.cols(); ++j) {
            std::vector<T> q_j = Q.getColumn(j);
            qt_b[j] = dotProduct(q_j, b);
        }

        std::cout << "Q^T * b = ";
        for (auto val : qt_b) std::cout << val << " ";
        std::cout << std::endl << std::endl;

        std::vector<T> x(R.cols(), 0.0);
        for (int i = R.cols() - 1; i >= 0; --i) {
            x[i] = qt_b[i];
            for (size_t j = i + 1; j < R.cols(); ++j) {
                x[i] -= R(i, j) * x[j];
            }
            x[i] /= R(i, i);
        }

        return x;
    }

    static std::vector<T> solveQR(const Matrix<T>& A, const std::vector<T>& b) {
        if (A.rows() != A.cols()) {
            throw std::invalid_argument("直接求解要求矩阵A为方阵");
        }
        if (A.rows() != b.size()) {
            throw std::invalid_argument("矩阵A和向量b维度不兼容");
        }

        return solveLeastSquares(A, b);
    }

    static std::pair<std::vector<T>, Matrix<T>> jacobiEigen(const Matrix<T>& A, size_t max_iter = 1000, T tolerance = 1e-10) {
        if (!A.isSymmetric()) {
            throw std::invalid_argument("雅可比方法要求对称矩阵");
        }

        size_t n = A.rows();
        Matrix<T> V(n, n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            V(i, i) = 1.0;
        }

        Matrix<T> B = A;

        for (size_t iter = 0; iter < max_iter; ++iter) {
            T max_off_diag = 0;
            size_t p = 0, q = 0;

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    T abs_val = std::abs(B(i, j));
                    if (abs_val > max_off_diag) {
                        max_off_diag = abs_val;
                        p = i;
                        q = j;
                    }
                }
            }

            if (max_off_diag < tolerance) {
                std::cout << "雅可比迭代在 " << iter << " 次后收敛" << std::endl;
                break;
            }

            T theta = (B(q, q) - B(p, p)) / (2 * B(p, q));
            T t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(1 + theta * theta));
            T c = 1.0 / std::sqrt(1 + t * t);
            T s = t * c;

            T B_pp = B(p, p);
            T B_qq = B(q, q);
            T B_pq = B(p, q);

            B(p, p) = c * c * B_pp - 2 * c * s * B_pq + s * s * B_qq;
            B(q, q) = s * s * B_pp + 2 * c * s * B_pq + c * c * B_qq;
            B(p, q) = B(q, p) = 0;

            for (size_t j = 0; j < n; ++j) {
                if (j != p && j != q) {
                    T B_pj = B(p, j);
                    T B_qj = B(q, j);
                    B(p, j) = B(j, p) = c * B_pj - s * B_qj;
                    B(q, j) = B(j, q) = s * B_pj + c * B_qj;
                }
            }

            for (size_t i = 0; i < n; ++i) {
                T V_ip = V(i, p);
                T V_iq = V(i, q);
                V(i, p) = c * V_ip - s * V_iq;
                V(i, q) = s * V_ip + c * V_iq;
            }
        }

        std::vector<T> eigenvalues(n);
        for (size_t i = 0; i < n; ++i) {
            eigenvalues[i] = B(i, i);
        }

        return {eigenvalues, V};
    }

    static void svdDecomposition(const Matrix<T>& A, Matrix<T>& U, std::vector<T>& S, Matrix<T>& Vt) {
        size_t m = A.rows();
        size_t n = A.cols();
        size_t k = std::min(m, n);

        std::cout << "开始 " << m << "x" << n << " 矩阵的SVD分解:" << std::endl;
        A.print("输入矩阵 A");

        Matrix<T> At = A.transpose();
        Matrix<T> AtA = multiply(At, A);

        std::cout << "A^T * A:" << std::endl;
        AtA.print();

        auto [eigenvalues, V] = jacobiEigen(AtA);

        std::cout << "A^T * A 的特征值: ";
        for (auto val : eigenvalues) std::cout << val << " ";
        std::cout << std::endl << std::endl;

        V.print("A^T * A 的特征向量 V");

        S.resize(k);
        for (size_t i = 0; i < k; ++i) {
            S[i] = std::sqrt(std::max(eigenvalues[i], T(0)));
        }

        std::vector<size_t> indices(k);
        for (size_t i = 0; i < k; ++i) indices[i] = i;
        std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
            return S[i] > S[j];
        });

        std::vector<T> S_sorted(k);
        Matrix<T> V_sorted(n, k, 0.0);

        for (size_t i = 0; i < k; ++i) {
            S_sorted[i] = S[indices[i]];
            for (size_t j = 0; j < n; ++j) {
                V_sorted(j, i) = V(j, indices[i]);
            }
        }
        S = S_sorted;
        Vt = V_sorted.transpose();

        std::cout << "排序后的奇异值: ";
        for (auto val : S) std::cout << val << " ";
        std::cout << std::endl << std::endl;

        Vt.print("V^T 矩阵 (按奇异值排序)");

        U = Matrix<T>(m, k, 0.0);
        for (size_t i = 0; i < k; ++i) {
            if (S[i] > 1e-10) {
                std::vector<T> u_col = A.getColumn(0);
                for (size_t j = 0; j < m; ++j) {
                    u_col[j] = 0;
                    for (size_t l = 0; l < n; ++l) {
                        u_col[j] += A(j, l) * V_sorted(l, i);
                    }
                    u_col[j] /= S[i];
                }
                U.setColumn(i, u_col);
            } else {
                std::vector<T> u_col(m, 0.0);
                u_col[i] = 1.0;
                U.setColumn(i, u_col);
            }
        }

        U.print("U 矩阵");

        Matrix<T> Sigma(m, n, 0.0);
        for (size_t i = 0; i < k; ++i) {
            Sigma(i, i) = S[i];
        }

        Matrix<T> USigma = multiply(U, Sigma);
        Matrix<T> USigmaVt = multiply(USigma, Vt);

        std::cout << "验证: U * Sigma * V^T 应等于原矩阵 A" << std::endl;
        USigmaVt.print("U * Sigma * V^T");

        T error = 0;
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                error += std::abs(USigmaVt(i, j) - A(i, j));
            }
        }
        std::cout << "重构误差: " << error << std::endl;
    }

    static std::vector<T> solveSVLeastSquares(const Matrix<T>& A, const std::vector<T>& b, T tolerance = 1e-10) {
        if (A.rows() != b.size()) {
            throw std::invalid_argument("矩阵A和向量b维度不兼容");
        }

        std::cout << "使用SVD求解最小二乘问题:" << std::endl;
        std::cout << "min ||Ax - b||?" << std::endl;

        size_t m = A.rows();
        size_t n = A.cols();

        Matrix<T> U(m, m, 0.0);
        std::vector<T> S;
        Matrix<T> Vt(n, n, 0.0);

        svdDecomposition(A, U, S, Vt);

        std::vector<T> ut_b(m, 0.0);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < m; ++j) {
                ut_b[i] += U(j, i) * b[j];
            }
        }

        std::cout << "U^T * b = ";
        for (auto val : ut_b) std::cout << val << " ";
        std::cout << std::endl << std::endl;

        std::vector<T> x(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            if (i < S.size() && S[i] > tolerance) {
                for (size_t j = 0; j < n; ++j) {
                    x[j] += Vt(i, j) * ut_b[i] / S[i];
                }
            }
        }

        return x;
    }

    static std::pair<T, std::vector<T>> powerIteration(const Matrix<T>& A, size_t max_iter = 1000, T tol = 1e-6) {
        if (A.rows() != A.cols()) {
            throw std::invalid_argument("幂迭代法要求矩阵为方阵");
        }

        size_t n = A.rows();
        std::vector<T> x(n, 1.0);
        T eigenvalue_old = 0.0;

        for (size_t iter = 0; iter < max_iter; ++iter) {
            std::vector<T> y = A * x;

            T eigenvalue = dotProduct(x, y) / dotProduct(x, x);

            x = normalize(y);

            if (std::abs(eigenvalue - eigenvalue_old) < tol) {
                std::cout << "幂迭代在 " << iter + 1 << " 次后收敛." << std::endl;
                return {eigenvalue, x};
            }

            eigenvalue_old = eigenvalue;
        }

        std::cout << "幂迭代在 " << max_iter << " 次内未收敛." << std::endl;
        return {eigenvalue_old, x};
    }

    static std::pair<T, std::vector<T>> inverseIteration(const Matrix<T>& A, size_t max_iter = 1000, T tol = 1e-6) {
        if (A.rows() != A.cols()) {
            throw std::invalid_argument("逆迭代法要求矩阵为方阵");
        }

        size_t n = A.rows();
        std::vector<T> x(n, 1.0);
        T eigenvalue_old = 0.0;

        for (size_t iter = 0; iter < max_iter; ++iter) {
            std::vector<T> y = solveLU(A, x);

            T eigenvalue = dotProduct(x, y) / dotProduct(x, x);

            x = normalize(y);

            if (std::abs(eigenvalue - eigenvalue_old) < tol) {
                std::cout << "逆迭代在 " << iter + 1 << " 次后收敛." << std::endl;
                return {1.0 / eigenvalue, x};
            }

            eigenvalue_old = eigenvalue;
        }

        std::cout << "逆迭代在 " << max_iter << " 次内未收敛." << std::endl;
        return {1.0 / eigenvalue_old, x};
    }
};

#endif // LINEAR_H
