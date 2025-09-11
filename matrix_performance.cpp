#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include "matrix_base.h"
#include "matrix_operations.h"
#include "matrix_decomposition.h"
#include "numerical_optimization.h"
#include "expression_templates.h"

using namespace matrixlib;
using namespace Eigen;
using namespace std;
using namespace chrono;

// 计时工具类
template <typename T>
struct Timer
{
    high_resolution_clock::time_point start;
    string name;
    double duration; // 存储耗时(毫秒)

    Timer(string n) : name(n)
    {
        start = high_resolution_clock::now();
    }

    ~Timer()
    {
        auto end = high_resolution_clock::now();
        duration = duration_cast<microseconds>(end - start).count() / 1000.0;
        cout << left << setw(50) << name << "耗时: " << fixed << setprecision(6) << duration << " ms" << endl;
    }
};

// 结果验证工具
template <typename T>
bool verify_result(const matrixlib::Matrix<T> &our_result, const MatrixXd &eigen_result, double epsilon = 1e-6)
{
    if (our_result.rows() != eigen_result.rows() || our_result.cols() != eigen_result.cols())
        return false;

    for (size_t i = 0; i < our_result.rows(); ++i)
    {
        for (size_t j = 0; j < our_result.cols(); ++j)
        {
            if (fabs(fabs(our_result(i, j)) - fabs(eigen_result(i, j))) > epsilon)
            {
                cerr << "结果不匹配 at (" << i << "," << j << "): 我们的库="
                     << our_result(i, j) << ", Eigen=" << eigen_result(i, j) << endl;
                return false;
            }
        }
    }
    return true;
}

template <typename T>
bool verify_scalar(T our_result, T eigen_result, double epsilon = 1e-6)
{
    if (fabs(our_result - eigen_result) > epsilon)
    {
        cerr << "标量结果不匹配: 我们的库=" << our_result << ", Eigen=" << eigen_result << endl;
        return false;
    }
    return true;
}

// 将自定义矩阵转换为Eigen矩阵
MatrixXd convert_to_eigen(const matrixlib::Matrix<double> &our_mat)
{
    MatrixXd eigen_mat(our_mat.rows(), our_mat.cols());
    for (size_t i = 0; i < our_mat.rows(); ++i)
    {
        for (size_t j = 0; j < our_mat.cols(); ++j)
        {
            eigen_mat(i, j) = our_mat(i, j); // 逐个元素复制
        }
    }
    return eigen_mat;
}

// 基础运算测试
void test_basic_operations(size_t size)
{
    cout << "\n=== 基础运算测试 (" << size << "x" << size << ") ===" << endl;

    // 生成测试矩阵
    auto our_mat1 = random_matrix<double>(size, size, -100, 100);
    auto our_mat2 = random_matrix<double>(size, size, -100, 100);
    auto our_vec = random_matrix<double>(size, 1, -100, 100);

    MatrixXd eigen_mat1 = MatrixXd::Random(size, size) * 100;
    MatrixXd eigen_mat2 = MatrixXd::Random(size, size) * 100;
    VectorXd eigen_vec = VectorXd::Random(size) * 100;

    // 矩阵加法
    {
        Timer<double> t("当前库: 矩阵加法");
        auto result = our_mat1 + our_mat2;
    }
    {
        Timer<double> t("Eigen: 矩阵加法");
        auto result = eigen_mat1 + eigen_mat2;
    }

    // 矩阵数乘
    {
        Timer<double> t("当前库: 矩阵数乘");
        auto result = our_mat1 * 2.5;
    }
    {
        Timer<double> t("Eigen: 矩阵数乘");
        auto result = eigen_mat1 * 2.5;
    }

    // 矩阵向量乘法
    {
        Timer<double> t("当前库: 矩阵向量乘法");
        auto result = our_mat1 * our_vec;
    }
    {
        Timer<double> t("Eigen: 矩阵向量乘法");
        auto result = eigen_mat1 * eigen_vec;
    }

    // 矩阵转置
    {
        Timer<double> t("当前库: 矩阵转置");
        auto result = transpose(our_mat1);
    }
    {
        Timer<double> t("Eigen: 矩阵转置");
        auto result = eigen_mat1.transpose();
    }
}

// 矩阵乘法详细测试
void test_matrix_multiplication(size_t size)
{
    cout << "\n=== 矩阵乘法测试 (" << size << "x" << size << ") ===" << endl;

    // 生成测试矩阵
    auto our_mat1 = random_matrix<double>(size, size, -100, 100);
    auto our_mat2 = random_matrix<double>(size, size, -100, 100);

    MatrixXd eigen_mat1 = convert_to_eigen(our_mat1);

    MatrixXd eigen_mat2 = convert_to_eigen(our_mat2);

    // 普通矩阵乘法
    {
        Timer<double> t("当前库: 普通矩阵乘法");
        auto our_result = our_mat1 * our_mat2;
        Timer<double> t_verify("验证: 当前库 vs Eigen 矩阵乘法");
        MatrixXd eigen_result = eigen_mat1 * eigen_mat2;
        bool valid = verify_result(our_result, eigen_result);
        cout << left << setw(50) << " " << "结果: " << (valid ? "正确" : "错误") << endl;
    }

    // Eigen矩阵乘法（单独计时，排除验证时间）
    {
        Timer<double> t("Eigen: 矩阵乘法");
        auto result = eigen_mat1 * eigen_mat2;
    }
}

// 矩阵分解测试
void test_decompositions(size_t size)
{
    cout << "\n=== 矩阵分解测试 (" << size << "x" << size << ") ===" << endl;

    // 生成对称正定矩阵
    auto our_mat = random_matrix<double>(size, size, -100, 100);
    auto our_sym_mat = our_mat * transpose(our_mat) + identity_matrix<double>(size) * 10.0;

    MatrixXd eigen_mat = convert_to_eigen(our_mat);
    MatrixXd eigen_sym_mat = convert_to_eigen(our_sym_mat);

    // LU分解
    {
        Timer<double> t("当前库: LU分解");
        auto [our_L, our_U] = lu_decomposition(our_mat);

        Timer<double> t_verify("验证: 当前库LU分解");
        PartialPivLU<MatrixXd> lu(eigen_mat);
        MatrixXd eigen_L = MatrixXd::Identity(size, size);
        eigen_L.triangularView<StrictlyLower>() = lu.matrixLU();
        MatrixXd eigen_U = lu.matrixLU().triangularView<Upper>();

        bool valid_L = verify_result(our_L, eigen_L, 1e-4);
        bool valid_U = verify_result(our_U, eigen_U, 1e-4);
        cout << left << setw(50) << " " << "结果: " << (valid_L && valid_U ? "正确" : "错误") << endl;
    }
    {
        Timer<double> t("Eigen: LU分解");
        PartialPivLU<MatrixXd> lu(eigen_mat);
    }

    // Cholesky分解
    {
        Timer<double> t("当前库: Cholesky分解");
        auto L = cholesky_decomposition(our_sym_mat);

        Timer<double> t_verify("验证: 当前库Cholesky分解");
        LLT<MatrixXd> llt(eigen_sym_mat);
        MatrixXd eigen_L = llt.matrixL();
        bool valid = verify_result(L, eigen_L);
        cout << left << setw(50) << " " << "结果: " << (valid ? "正确" : "错误") << endl;
    }
    {
        Timer<double> t("Eigen: Cholesky分解");
        LLT<MatrixXd> llt(eigen_sym_mat);
    }

    // QR分解
    {
        Timer<double> t("当前库: QR分解");
        auto [our_Q, our_R] = qr_decomposition(our_mat);

        Timer<double> t_verify("验证: 当前库QR分解");
        HouseholderQR<MatrixXd> qr(eigen_mat);
        MatrixXd eigen_Q = qr.householderQ();
        MatrixXd QR = qr.matrixQR();
        MatrixXd eigen_R = QR.triangularView<Upper>();
        // 截取Eigen的Q矩阵前size列（与our_Q维度匹配）
        eigen_Q = eigen_Q.leftCols(size);
        bool valid_Q = verify_result(our_Q, eigen_Q, 1e-4); // QR分解符号可能有差异，放宽精度
        bool valid_R = verify_result(our_R, eigen_R, 1e-4);
        cout << left << setw(50) << " " << "结果: " << (valid_Q && valid_R ? "正确" : "错误") << endl;
    }

    {
        Timer<double> t("Eigen: QR分解");
        HouseholderQR<MatrixXd> qr(eigen_mat);
    }
}

// 线性方程组求解测试
void test_linear_solvers(size_t size)
{
    cout << "\n=== 线性方程组求解测试 (" << size << "x" << size << ") ===" << endl;

    // 生成线性方程组 Ax = b
    auto our_A = random_matrix<double>(size, size, -100, 100);
    auto our_x_true = random_matrix<double>(size, 1, -100, 100);
    auto our_b = our_A * our_x_true;

    MatrixXd eigen_A = convert_to_eigen(our_A);
    VectorXd eigen_x_true = convert_to_eigen(our_x_true);
    VectorXd eigen_b = eigen_A * eigen_x_true;

    // 共轭梯度法
    {
        Timer<double> t("当前库: 共轭梯度法求解");
        auto our_x = conjugate_gradient(our_A * transpose(our_A), our_b);

        Timer<double> t_op("当前库: 共轭梯度法求解并使用表达式模板");
        auto our_x_op = conjugate_gradient_op(our_A * transpose(our_A), our_b);

        Timer<double> t_verify("验证: 共轭梯度法求解结果");
        ConjugateGradient<MatrixXd> cg;
        cg.compute(eigen_A * eigen_A.transpose());
        VectorXd eigen_x = cg.solve(eigen_b);
        bool valid = verify_result(our_x, eigen_x, 1e-4);
        cout << left << setw(50) << " " << "结果: " << (valid ? "正确" : "错误") << endl;
    }

    {
        Timer<double> t("Eigen: 共轭梯度法求解");
        ConjugateGradient<MatrixXd> cg;
        cg.compute(eigen_A * eigen_A.transpose());
        auto x = cg.solve(eigen_b);
    }

    // 雅可比迭代
    {
        Timer<double> t("当前库: 雅可比迭代求解");
        auto our_x = jacobi_iteration(our_A, our_b);

        Timer<double> t_verify("验证: 雅可比迭代结果");
        VectorXd eigen_x = eigen_A.colPivHouseholderQr().solve(eigen_b); // 用Eigen精确解验证
        bool valid = verify_result(our_x, eigen_x, 1e-3);                // 迭代法精度较低
        cout << left << setw(50) << " " << "结果: " << (valid ? "正确" : "错误") << endl;
    }

    // 高斯-赛德尔迭代
    {
        Timer<double> t("当前库: 高斯-赛德尔迭代求解");
        auto our_x = gauss_seidel_iteration(our_A, our_b);

        Timer<double> t_verify("验证: 高斯-赛德尔迭代结果");
        VectorXd eigen_x = eigen_A.colPivHouseholderQr().solve(eigen_b);
        bool valid = verify_result(our_x, eigen_x, 1e-3);
        cout << left << setw(50) << " " << "结果: " << (valid ? "正确" : "错误") << endl;
    }
}

// 行列式与矩阵求逆测试
void test_determinant_inverse(size_t size)
{
    cout << "\n=== 行列式与矩阵求逆测试 (" << size << "x" << size << ") ===" << endl;

    // 生成可逆矩阵
    auto our_mat = random_matrix<double>(size, size, -100, 100);
    our_mat = our_mat * transpose(our_mat) + identity_matrix<double>(size) * 10.0; // 保证可逆
    MatrixXd eigen_mat = convert_to_eigen(our_mat);

    // 行列式计算
    {
        Timer<double> t("当前库: 行列式计算");
        double our_det = determinant(our_mat);

        Timer<double> t_verify("验证: 行列式结果");
        double eigen_det = eigen_mat.determinant();
        bool valid = verify_scalar(our_det, eigen_det, 100);
        cout << left << setw(50) << " " << "结果: " << (valid ? "正确" : "错误") << endl;
    }
    {
        Timer<double> t("Eigen: 行列式计算");
        double eigen_det = eigen_mat.determinant();
    }

    // 矩阵求逆
    {
        Timer<double> t("当前库: 矩阵求逆");
        auto our_inv = inverse(our_mat);

        Timer<double> t_verify("验证: 矩阵求逆结果");
        MatrixXd eigen_inv = eigen_mat.inverse();
        bool valid = verify_result(our_inv, eigen_inv, 1e-3);
        cout << left << setw(50) << " " << "结果: " << (valid ? "正确" : "错误") << endl;
    }
    {
        Timer<double> t("Eigen: 矩阵求逆");
        MatrixXd eigen_inv = eigen_mat.inverse();
    }
}

// 最小二乘法与SVD测试（新增）
void test_least_squares_svd(size_t rows, size_t cols)
{
    cout << "\n=== 最小二乘法与SVD测试 (" << rows << "x" << cols << ") ===" << endl;

    // 生成超定方程组（rows > cols）
    auto our_A = random_matrix<double>(rows, cols, -100, 100);
    auto our_b = random_matrix<double>(rows, 1, -100, 100);
    MatrixXd eigen_A = convert_to_eigen(our_A);
    VectorXd eigen_b = convert_to_eigen(our_b);

    // 最小二乘法
    {
        Timer<double> t("当前库: 最小二乘法");
        auto our_x = least_squares(our_A, our_b);

        Timer<double> t_verify("验证: 最小二乘法结果");
        VectorXd eigen_x = eigen_A.jacobiSvd(ComputeThinU | ComputeThinV).solve(eigen_b);
        bool valid = verify_result(our_x, eigen_x, 1e-3);
        cout << left << setw(50) << " " << "结果: " << (valid ? "正确" : "错误") << endl;
    }
    {
        Timer<double> t("Eigen: 最小二乘法（SVD）");
        VectorXd eigen_x = eigen_A.jacobiSvd(ComputeThinU | ComputeThinV).solve(eigen_b);
    }

    // SVD分解（仅测试方阵，简化对比）
    if (rows == cols)
    {
        {
            Timer<double> t("当前库: SVD分解");
            auto [our_U, our_Sigma, our_V] = svd(our_A);

            Timer<double> t_verify("验证: SVD分解结果");
            JacobiSVD<MatrixXd> svd(eigen_A, ComputeThinU | ComputeThinV);
            MatrixXd eigen_U = svd.matrixU();
            MatrixXd eigen_Sigma = svd.singularValues().asDiagonal();
            MatrixXd eigen_V = svd.matrixV();

            matrixlib::Matrix our_recon = our_U * our_Sigma * transpose(our_V);
            MatrixXd eigen_recon = eigen_U * eigen_Sigma * eigen_V.transpose();
            bool valid = verify_result(our_recon, eigen_recon, 1e-2); // SVD精度要求较低
            cout << left << setw(50) << " " << "结果: " << (valid ? "正确" : "错误") << endl;
        }
        {
            Timer<double> t("Eigen: SVD分解");
            JacobiSVD<MatrixXd> svd(eigen_A, ComputeThinU | ComputeThinV);
        }
    }
}

int main()
{
    cout << "=== MatrixLib与Eigen性能对比测试 ===" << endl;
    cout << "注意: 所有测试结果包含运算时间和正确性验证" << endl
         << endl;

    // 测试不同规模矩阵
    vector<size_t> sizes = {4, 8, 32, 64, 128, 256, 512};

    for (auto s : sizes)
    {
        if (s <= 8)
        {
            test_determinant_inverse(s);
        }
        else
        {
            test_basic_operations(s);
            test_matrix_multiplication(s);
            test_decompositions(s);
            test_linear_solvers(s);

            test_least_squares_svd(s, s);
        }

        cout << string(80, '-') << endl;
    }

    return 0;
}
