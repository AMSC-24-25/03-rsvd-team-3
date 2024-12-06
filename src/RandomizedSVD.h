#ifndef RANDOMIZEDSVD_H
#define RANDOMIZEDSVD_H

#include <Eigen/Core>
#include <Eigen/Dense>

namespace Eigen {

/**
 * @brief A class to perform Randomized Singular Value Decomposition (RSVD)
 */
template<typename MatrixType, int Options = ComputeThinU | ComputeThinV>
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
    RandomizedSVD& compute(const MatrixType& matrix, Index rank, Index powerIterations = 2) {
        randomProjection(matrix, rank, powerIterations);
        return *this;
    }

    /**
     * @brief Get the singular values
     * @return A vector containing the singular values
     */
    const DenseVector& singularValues() const { return m_singularValues; }

    /**
     * @brief Get the left singular vectors (U matrix)
     * @return The U matrix
     */
    const DenseMatrix& matrixU() const { return m_matrixU; }

    /**
     * @brief Get the right singular vectors (V matrix)
     * @return The V matrix
     */
    const DenseMatrix& matrixV() const { return m_matrixV; }

private:
    /**
     * @brief Perform the randomized projection
     *
     * @param matrix The input matrix
     * @param rank Desired rank of the approximation
     * @param powerIterations Number of power iterations
     */
    void randomProjection(const MatrixType& matrix, Index rank, Index powerIterations) {
        // Step 1: Generate a random Gaussian matrix
        DenseMatrix randomMatrix = DenseMatrix::Random(matrix.cols(), rank);

        // Step 2: Form the sketch matrix
        DenseMatrix sketch = matrix * randomMatrix;

        // Step 3: Perform power iterations to refine the sketch
        for (Index i = 0; i < powerIterations; ++i) {
            sketch = matrix * (matrix.transpose() * sketch);
        }

        // Step 4: Compute the QR decomposition of the sketch
        HouseholderQR<DenseMatrix> qr(sketch);
        DenseMatrix Q = qr.householderQ();

        // Step 5: Project the original matrix onto the low-dimensional subspace
        DenseMatrix B = Q.transpose() * matrix;

        // Step 6: Perform SVD on the small matrix B
        JacobiSVD<DenseMatrix> svd(B, Options);
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

#endif //RANDOMIZEDSVD_H
