#ifndef GIVENSROTATIONQR_H
#define GIVENSROTATIONQR_H

#include "MatrixTraits.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <omp.h>

namespace Eigen {
    /**
     * @brief Base class template (default for dense matrices)
     */
    template<typename MatrixType, bool IsSparse = is_sparse_matrix<MatrixType>::value>
    class GivensRotationQR {
    public:
        using Scalar = typename MatrixType::Scalar;
        using Index = typename MatrixType::Index;
        static constexpr int StorageOrder = MatrixType::IsRowMajor ? RowMajor : ColMajor;
        using DenseMat = Matrix<Scalar, Dynamic, Dynamic>;

        GivensRotationQR() = default;

        /**
         * @brief Compute the QR decomposition using Given rotations
         *
         * @param matrix The input matrix to decompose
         * @return Reference to this object
         */
        GivensRotationQR& compute(const MatrixType& matrix) {
            Index rows = matrix.rows();
            Index cols = matrix.cols();
            m_matrixR = matrix;
            m_matrixQ = DenseMat::Identity(rows, rows);

            for (Index j = 0; j < cols; ++j) {
                for (Index i = rows - 1; i > j; --i) {
                    Scalar a = m_matrixR(i - 1, j);
                    Scalar b = m_matrixR(i, j);
                    if (std::abs(b) < 1e-9) {
                        continue;
                    }

                    Scalar c, s;
                    computeGivensRotation(a, b, c, s);
                    applyGivensRotationQ(m_matrixQ, i - 1, i, c, s);
                    applyGivensRotationR(m_matrixR, i - 1, i, c, s);
                }
            }

            return *this;
        }

        /**
         * @brief Access the orthogonal matrix Q
         * @return The Q matrix
         */
        const DenseMat& matrixQ() const { return m_matrixQ; }

        /**
         * @brief Access the upper triangular matrix R
         * @return The R matrix
         */
        const DenseMat& matrixR() const { return m_matrixR; }

    private:
        /**
         * @brief Compute the Given rotation coefficients
         *
         * @param a First value
         * @param b Second value
         * @param c Output cosine
         * @param s Output sine
         */
        void computeGivensRotation(Scalar a, Scalar b, Scalar& c, Scalar& s) {
            if (b == Scalar(0)) {
                c = Scalar(1);
                s = Scalar(0);
            } else {
                Scalar r = std::hypot(a, b);
                c = a / r;
                s = -b / r;
            }
        }
        /**
         * @brief Apply a Given rotation to a matrix, construct Q
         *
         * @param matrix The matrix to apply the rotation to
         * @param i Row index 1
         * @param k Row index 2
         * @param c Cosine of the rotation
         * @param s Sine of the rotation
         */
        void applyGivensRotationQ(DenseMat& matrix, Index i, Index k, Scalar c, Scalar s) {
            /*
             * Apply the rotation to the columns of the matrix
             * Q = I * G_1^T * G_2^T * ... * G_n^T
             * Instead of computing the transpose of the rotation matrix, we apply the rotation to the columns
             * QT = G_1 * G_2 * ... * G_n * I
             */
            auto temp_i = matrix.col(i).eval();
            auto temp_k = matrix.col(k).eval();
            matrix.col(i) = c * temp_i - s * temp_k;
            matrix.col(k) = s * temp_i + c * temp_k;
        }

        /**
         * @brief Apply a Given rotation to a matrix, construct R
         *
         * @param matrix The matrix to apply the rotation to
         * @param i Row index 1
         * @param k Row index 2
         * @param c Cosine of the rotation
         * @param s Sine of the rotation
         */
        void applyGivensRotationR(DenseMat& matrix, Index i, Index k, Scalar c, Scalar s) {
                /*
                 * Apply the rotation to the rows of the matrix
                 * R = G_n * ... * G_2 * G_1 * A
                 */
                auto temp_i = matrix.row(i).eval();
                auto temp_k = matrix.row(k).eval();
                matrix.row(i) = c * temp_i - s * temp_k;
                matrix.row(k) = s * temp_i + c * temp_k;
        }

        DenseMat m_matrixQ;
        DenseMat m_matrixR;
    };

    /**
     * @brief sparse matrices
     */
    template<typename MatrixType>
    class GivensRotationQR<MatrixType, true> {
    public:
        using Scalar = typename MatrixType::Scalar;
        using Index = typename MatrixType::Index;
        static constexpr int StorageOrder = MatrixType::IsRowMajor ? RowMajor : ColMajor;
        using DenseMat = Matrix<Scalar, Dynamic, Dynamic, StorageOrder>;
        using DenseVector = Matrix<Scalar, Dynamic, 1>;
        using SparseMat = SparseMatrix<Scalar, StorageOrder, Index>;

        GivensRotationQR() = default;

        GivensRotationQR& compute(const MatrixType& matrix) {
            Index rows = matrix.rows();
            Index cols = matrix.cols();
            m_matrixR = DenseMat(matrix);
            m_matrixQ = DenseMat::Identity(rows, rows);

            for (Index j = 0; j < cols; ++j) {
                // find the column range
                Index col_start = matrix.outerIndexPtr()[j];
                Index col_end   = matrix.outerIndexPtr()[j + 1];

                // collect all non-zero rows in the column
                std::vector<Index> rowList;
                rowList.reserve(col_end - col_start);

                for (Index p = col_start; p < col_end; ++p) {
                    Index r = matrix.innerIndexPtr()[p];
                    // Only concerned with the lower triangle (r > j)
                    if (r > j) {
                        rowList.push_back(r);
                    }
                }
                // Sort in descending order
                std::sort(rowList.begin(), rowList.end(), std::greater<Index>());

                // Execute Givens in the dense m_matrixR
                for (Index r : rowList) {
                    Scalar a = m_matrixR(r-1, j);
                    Scalar b = m_matrixR(r,   j);
                    if (std::abs(b) < Scalar(1e-15)) {
                        continue;
                    }

                    Scalar c, s;
                    computeGivensRotation(a, b, c, s);
                    applyGivensRotationQ(m_matrixQ, r-1, r, c, s);
                    applyGivensRotationR(m_matrixR, r-1, r, c, s);
                }
            }

            return *this;
        }

        /**
         * @brief Access the orthogonal matrix Q
         * @return The Q matrix
         */
        const DenseMat& matrixQ() const { return m_matrixQ; }

        /**
         * @brief Access the upper triangular matrix R
         * @return The R matrix
         */
        const DenseMat& matrixR() const { return m_matrixR; }

    private:
        /**
         * @brief Compute the Given rotation coefficients
         *
         * @param a First value
         * @param b Second value
         * @param c Output cosine
         * @param s Output sine
         */
        void computeGivensRotation(Scalar a, Scalar b, Scalar& c, Scalar& s) {
            if (b == Scalar(0)) {
                c = Scalar(1);
                s = Scalar(0);
            } else {
                Scalar r = std::hypot(a, b);
                c = a / r;
                s = -b / r;
            }
        }

        /**
         * @brief Apply a Given rotation to a matrix, construct Q
         *
         * @param matrix The matrix to apply the rotation to
         * @param i Row index 1
         * @param k Row index 2
         * @param c Cosine of the rotation
         * @param s Sine of the rotation
         */
        void applyGivensRotationQ(DenseMat& matrix, Index i, Index k, Scalar c, Scalar s) {
            /*
             * Apply the rotation to the columns of the matrix
             * Q = I * G_1^T * G_2^T * ... * G_n^T
             * Instead of computing the transpose of the rotation matrix, we apply the rotation to the columns
             * QT = G_1 * G_2 * ... * G_n * I
             */
            const Index rows = matrix.rows();
            const Index cols = matrix.cols();
            Scalar* data_ptr = matrix.data();

            if constexpr (StorageOrder == RowMajor) {
                // RowMajor: (row, col) => row*cols + col
                for (Index row = 0; row < rows; ++row) {
                    const Index offset_i = row * cols + i;
                    const Index offset_k = row * cols + k;
                    Scalar temp_i = data_ptr[offset_i];
                    Scalar temp_k = data_ptr[offset_k];
                    data_ptr[offset_i] = c * temp_i - s * temp_k;
                    data_ptr[offset_k] = s * temp_i + c * temp_k;
                }
            } else {
                // ColMajor: (row, col) => col*rows + row
                for (Index row = 0; row < rows; ++row) {
                    const Index offset_i = i * rows + row;
                    const Index offset_k = k * rows + row;
                    Scalar temp_i = data_ptr[offset_i];
                    Scalar temp_k = data_ptr[offset_k];
                    data_ptr[offset_i] = c * temp_i - s * temp_k;
                    data_ptr[offset_k] = s * temp_i + c * temp_k;
                }
            }
        }

        /**
         * @brief Apply a Given rotation to a matrix, construct R
         *
         * @param matrix The matrix to apply the rotation to
         * @param i Row index 1
         * @param k Row index 2
         * @param c Cosine of the rotation
         * @param s Sine of the rotation
         */
        void applyGivensRotationR(DenseMat& matrix, Index i, Index k, Scalar c, Scalar s) {
            /*
             * Apply the rotation to the rows of the matrix
             * R = G_n * ... * G_2 * G_1 * A
             */
            const Index rows = matrix.rows();
            const Index cols = matrix.cols();
            Scalar* data_ptr = matrix.data();

            if constexpr (StorageOrder == RowMajor) {
                // RowMajor: (row, col) => row*cols + col
                Scalar* row_i_ptr = data_ptr + i * cols;
                Scalar* row_k_ptr = data_ptr + k * cols;
                for (Index col = 0; col < cols; ++col) {
                    // ReSharper disable once CppDFANullDereference
                    Scalar temp_i = row_i_ptr[col];
                    Scalar temp_k = row_k_ptr[col];
                    row_i_ptr[col] = c * temp_i - s * temp_k;
                    row_k_ptr[col] = s * temp_i + c * temp_k;
                }
            } else {
                // ColMajor: (row, col) => col*rows + row
                //           row_i => col*rows + row_i
                //           row_k => col*rows + row_k
                for (Index col = 0; col < cols; ++col) {
                    const Index offset_i = col * rows + i;
                    const Index offset_k = col * rows + k;
                    Scalar temp_i = data_ptr[offset_i];
                    Scalar temp_k = data_ptr[offset_k];
                    data_ptr[offset_i] = c * temp_i - s * temp_k;
                    data_ptr[offset_k] = s * temp_i + c * temp_k;
                }
            }
        }

        DenseMat m_matrixQ;
        DenseMat m_matrixR;
    };

} // namespace Eigen

#endif // GIVENSROTATIONQR_H