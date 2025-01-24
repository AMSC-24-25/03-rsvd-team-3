#ifndef RANDOMIZEDSVD_H
#define RANDOMIZEDSVD_H

#include "MatrixTraits.h"
#include "PowerMethodSVD.h"
#include "GivensRotationQR.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <random>

namespace Eigen {

    /**
     * @brief A class to perform Randomized Singular Value Decomposition (RSVD), default for dense matrices
     */
    template<typename MatrixType, bool IsSparse = is_sparse_matrix<MatrixType>::value>
    class RandomizedSVD {
    public:
        using Scalar = typename MatrixType::Scalar;
        using Index = typename MatrixType::Index;
        using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
        using DenseVector = Matrix<Scalar, Dynamic, 1>;

        /**
         * @brief Default constructor
         */
        RandomizedSVD() = default;

        /**
         * @brief Compute the RSVD of a matrix
         *
         * @param matrix The input matrix to decompose
         * @param rank Desired rank of the approximation
         * @param powerIterations Number of power iterations for accuracy (default: 2)
         * @return Reference to this RSVD object
         */
        RandomizedSVD &compute(const MatrixType &matrix, Index rank, Index powerIterations = 2) {
            randomProjection(matrix, rank, powerIterations);
            return *this;
        }

        /**
         * @brief Get the singular values
         * @return A vector containing the singular values
         */
        const DenseVector &singularValues() const { return m_singularValues; }

        /**
         * @brief Get the left singular vectors (U matrix)
         * @return The U matrix
         */
        const DenseMatrix &matrixU() const { return m_matrixU; }

        /**
         * @brief Get the right singular vectors (V matrix)
         * @return The V matrix
         */
        const DenseMatrix &matrixV() const { return m_matrixV; }

    private:
        /**
         * @brief Generate a random matrix
         *
         * @param rows Number of rows
         * @param cols Number of columns
         * @return A random matrix
         */
        DenseMatrix generateRandomMatrix(Index rows, Index cols) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<Scalar> dist(0.0, 1.0);

            DenseMatrix mat(rows, cols);
            for (Index i = 0; i < rows; ++i) {
                for (Index j = 0; j < cols; ++j) {
                    mat(i, j) = dist(gen);
                }
            }
            return mat;
        }

        /**
         * @brief Perform the randomized projection
         *
         * @param matrix The input matrix
         * @param rank Desired rank of the approximation
         * @param powerIterations Number of power iterations
         */
        void randomProjection(const MatrixType &matrix, Index rank, Index powerIterations) {
            // Step 1: Generate a random Gaussian matrix
            DenseMatrix randomMatrix = generateRandomMatrix(matrix.cols(), rank);

            // Step 2: Form the sketch matrix
            DenseMatrix sketch = matrix * randomMatrix;

            // Step 3: Perform power iterations to refine the sketch
            for (Index i = 0; i < powerIterations; ++i) {
                sketch = matrix * (matrix.transpose() * sketch);
            }

            // Step 4: Compute the QR decomposition of the sketch
            HouseholderQR<DenseMatrix> qr(sketch);
            DenseMatrix Q = qr.householderQ() * DenseMatrix::Identity(sketch.rows(), rank);

            // Step 5: Project the original matrix onto the low-dimensional subspace
            DenseMatrix B = Q.transpose() * matrix;

            // Step 6: Perform SVD on the small matrix B
            // PowerMethodSVD<DenseMatrix> svd;
            // svd.compute(B, rank, 1000, Scalar(1e-6));
            JacobiSVD<DenseMatrix> svd;
            svd.compute(B, ComputeThinU | ComputeThinV);
            m_singularValues = svd.singularValues();
            m_matrixU = Q * svd.matrixU();
            m_matrixV = svd.matrixV();
        }

        // Member variables to store results
        DenseMatrix m_matrixU;
        DenseMatrix m_matrixV;
        DenseVector m_singularValues;
    };


    // Specialization for Sparse Matrices
    template<typename MatrixType>
    class RandomizedSVD<MatrixType, true> {
    public:
        using Scalar = typename MatrixType::Scalar;
        using Index = typename MatrixType::Index;
        using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
        using DenseVector = Matrix<Scalar, Dynamic, 1>;

        static constexpr int StorageOrder = MatrixType::IsRowMajor ? RowMajor : ColMajor;

        using SparseMatrixType = SparseMatrix<Scalar, StorageOrder, Index>;

        /**
         * @brief Default constructor
         */
        RandomizedSVD() = default;

        /**
         * @brief Compute the RSVD of a sparse matrix
         *
         * @param matrix The input sparse matrix to decompose
         * @param rank Desired rank of the approximation
         * @param powerIterations Number of power iterations for accuracy (default: 2)
         * @return Reference to this RSVD object
         */
        RandomizedSVD &compute(const SparseMatrixType &matrix, Index rank, Index powerIterations = 2) {
            randomProjection(matrix, rank, powerIterations);
            return *this;
        }

        /**
         * @brief Get the singular values
         * @return A vector containing the singular values
         */
        const DenseVector &singularValues() const { return m_singularValues; }

        /**
         * @brief Get the left singular vectors (U matrix)
         * @return The U matrix
         */
        const DenseMatrix &matrixU() const { return m_matrixU; }

        /**
         * @brief Get the right singular vectors (V matrix)
         * @return The V matrix
         */
        const DenseMatrix &matrixV() const { return m_matrixV; }

    private:
        /**
         * @brief Generate a random Gaussian dense matrix
         *
         * @param rows Number of rows
         * @param cols Number of columns
         * @return A random dense matrix with entries drawn from a standard normal distribution
         */
        DenseMatrix generateRandomMatrix(Index rows, Index cols) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<Scalar> dist(0.0, 1.0);

            DenseMatrix mat(rows, cols);
            for (Index i = 0; i < rows; ++i) {
                for (Index j = 0; j < cols; ++j) {
                    mat(i, j) = dist(gen);
                }
            }
            return mat;
        }

        /**
         * @brief Perform the randomized projection for sparse matrices
         *
         * @param matrix The input sparse matrix
         * @param rank Desired rank of the approximation
         * @param powerIterations Number of power iterations
         */
        void randomProjection(const SparseMatrixType &matrix, Index rank, Index powerIterations) {
            // Step 1: Generate a random Gaussian dense matrix
            DenseMatrix randomMatrix = generateRandomMatrix(matrix.cols(), rank);
            // Step 2: Form the initial sketch matrix (Q = A * Omega)
            SparseMatrixType sketch = (matrix * randomMatrix).sparseView();
            // Step 3: Perform power iterations to enhance the approximation
            for (Index i = 0; i < powerIterations; ++i) {
                // Multiply by A^T * A
                sketch = matrix * (matrix.transpose() * sketch);
            }
            // Step 4: Compute the QR decomposition of the sketch
            GivensRotationQR<SparseMatrixType> qr;
            qr.compute(sketch);
            DenseMatrix Q = qr.matrixQ() * DenseMatrix::Identity(sketch.rows(), rank);

            // Step 5: Project the original matrix onto the low-dimensional subspace (B = Q^T * A)
            SparseMatrixType B = (Q.transpose() * matrix).sparseView();

            // Step 6: Perform SVD on the small dense matrix B
            PowerMethodSVD<SparseMatrixType> svd;
            svd.compute(B, rank, 1000, Scalar(1e-6));
            m_singularValues = svd.singularValues();
            m_matrixU = Q * svd.matrixU();
            m_matrixV = svd.matrixV();
        }

        // Member variables to store results
        DenseMatrix m_matrixU;
        DenseMatrix m_matrixV;
        DenseVector m_singularValues;
    };


} // namespace Eigen

#endif // RANDOMIZEDSVD_H