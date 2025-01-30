#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <stdexcept>
#include <Eigen/QR>
#include <vector>


using namespace Eigen;
using namespace std;

// Utility function to generate a random Gaussian matrix
MatrixXd generateRandomMatrix(int rows, int cols) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d(0, 1);

    MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat(i, j) = d(gen);

    return mat;
}

// Funzione per calcolare la pseudoinversa
MatrixXd pseudoInverse(const MatrixXd& matrix) {
    JacobiSVD<MatrixXd> svd(matrix, ComputeThinU | ComputeThinV);
    double tolerance = numeric_limits<double>::epsilon() * max(matrix.cols(), matrix.rows()) * svd.singularValues().array().abs()(0);
    return svd.matrixV() * (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
}

// Algorithm 2: Subspace Iterations
MatrixXd sub_iterations(const MatrixXd &A, MatrixXd Y, int q) {
    for (int j = 0; j < q; ++j) {
        HouseholderQR<MatrixXd> qr1(Y);
        Y = qr1.householderQ();

        if (A.rows() != Y.rows()) {
            throw invalid_argument("Dimension mismatch: A.cols() != Y.rows()");
        }
        Y = A.transpose() * Y;

        HouseholderQR<MatrixXd> qr2(Y);
        Y = qr2.householderQ();

        if (A.cols() != Y.rows()) {
            throw invalid_argument("Dimension mismatch: A.rows() != Y.rows()");
        }
        Y = A * Y;

    }
    return Y;
}

// Algorithm 4: Randomized QB Decomposition
pair<MatrixXd, MatrixXd> rqb(const MatrixXd &A, int k, int p, int q) {
    int l = k + p;

    // Step 2: Generate random matrix
    MatrixXd Omega = generateRandomMatrix(A.cols(), l);

    // Step 3: Compute sketch
    MatrixXd Y = A * Omega;

    // Step 4: Subspace iterations (optional)
    Y = sub_iterations(A, Y, q);

    // Step 9: QR Decomposition
    HouseholderQR<MatrixXd> qr(Y);
    MatrixXd Q_full = qr.householderQ(); // Retrieve the full Q matrix
    MatrixXd Q = Q_full.leftCols(k);    // Extract the first k columns

    // Step 10: Project to low-dimensional space
    MatrixXd B = Q.transpose() * A;

    return {Q, B};
}

// Algorithm 9: Deterministic ID for Randomized CUR
pair<MatrixXd, VectorXi> id(const MatrixXd &A, int k) {
    // Step 1: Perform pivoted QR decomposition
    ColPivHouseholderQR<MatrixXd> qr(A);
    MatrixXd R = qr.matrixQR().triangularView<Upper>();
    VectorXi P = qr.colsPermutation().indices();

    // Step 2: Compute the pseudoinverse of S(1:k, 1:k)
    MatrixXd S_k = R.block(0, 0, k, k);
    MatrixXd S_k_pinv = S_k.completeOrthogonalDecomposition().pseudoInverse();

    // Step 3: Compute expansion coefficients
    MatrixXd T = S_k_pinv * R.block(0, k, k, A.cols() - k);

    // Step 4: Create an empty k x n matrix Z
    MatrixXd Z = MatrixXd::Zero(k, A.cols());

    // Step 5: Fill Z with ordered expansion coefficients using pivots P
    Z.block(0, 0, k, k) = MatrixXd::Identity(k, k);
    Z.block(0, k, k, A.cols() - k) = T;

    // Apply permutation to Z
    MatrixXd Z_permuted = MatrixXd::Zero(k, A.cols());
    for (int i = 0; i < A.cols(); ++i) {
        Z_permuted.col(P[i]) = Z.col(i);
    }

    // Step 6: Extract top-k column indices from pivots
    VectorXi J = P.head(k);

    // Step 7: Extract k columns from the input matrix A
    MatrixXd C(A.rows(), J.size());
    for (int i = 0; i < J.size(); ++i) {
        C.col(i) = A.col(J(i)); // Select each column in J
    }

    return {Z_permuted, J};
}

// Algorithm 9: Deterministic ID
tuple<MatrixXd, MatrixXd, VectorXi> id_determ(const MatrixXd &A, int k) {
    // Step 1: Perform pivoted QR decomposition
    ColPivHouseholderQR<MatrixXd> qr(A);
    MatrixXd R = qr.matrixQR().triangularView<Upper>();
    VectorXi P = qr.colsPermutation().indices();

    // Step 2: Compute the pseudoinverse of S(1:k, 1:k)
    MatrixXd S_k = R.block(0, 0, k, k);
    MatrixXd S_k_pinv = S_k.completeOrthogonalDecomposition().pseudoInverse();

    // Step 3: Compute expansion coefficients
    MatrixXd T = S_k_pinv * R.block(0, k, k, A.cols() - k);

    // Step 4: Create an empty k x n matrix Z
    MatrixXd Z = MatrixXd::Zero(k, A.cols());

    // Step 5: Fill Z with ordered expansion coefficients using pivots P
    Z.block(0, 0, k, k) = MatrixXd::Identity(k, k);
    Z.block(0, k, k, A.cols() - k) = T;

    // Apply permutation to Z
    MatrixXd Z_permuted = MatrixXd::Zero(k, A.cols());
    for (int i = 0; i < A.cols(); ++i) {
        Z_permuted.col(P[i]) = Z.col(i);
    }

    // Step 6: Extract top-k column indices from pivots
    VectorXi J = P.head(k);

    // Step 7: Extract k columns from the input matrix A
    // MatrixXd C = A.block(0, 0, A.rows(), k);
    MatrixXd C(A.rows(), J.size());
    for (int i = 0; i < J.size(); ++i) {
        C.col(i) = A.col(J(i)); // Select each column in J
    }

    return {C, Z_permuted, J};
}

// Algorithm 10: Randomized ID
tuple<MatrixXd, MatrixXd, VectorXi> rid(const MatrixXd &A, int k, int p, int q) {
    auto [Q, B] = rqb(A, k, p, q);
    auto [Z, J] = id(B, k); 
    
    MatrixXd C(A.rows(), J.size());
    for (int i = 0; i < J.size(); ++i) {
        C.col(i) = A.col(J(i)); // Select each column in J
    }
    return {C, Z, J};
}

// Algorithm 8: Randomized CUR Decomposition
tuple<MatrixXd, MatrixXd, MatrixXd> rcur(const MatrixXd &A, int k, int p, int q, bool rand) {
    MatrixXd C, Z;
    VectorXi J;

    // Selection between Randomized and Deterministic algorithm
    if(rand) {
        // (1) Randomized Column ID (using rid e id)
        tie(C, Z, J) = rid(A, k, p, q);
    } else {
        // (2) Deterministic Column ID
        tie(C, Z, J) = id_determ(A, k);
    }

    if (C.cols() != k) {
        throw invalid_argument("Dimension mismatch: C.cols() != k");
    }

    ColPivHouseholderQR<MatrixXd> qr(C.transpose());
    MatrixXd R = qr.matrixQR().triangularView<Upper>();

    VectorXi I = qr.colsPermutation().indices().head(k);
    MatrixXd R_mat(I.size(), A.cols());
    for (int i = 0; i < I.size(); ++i) {
        R_mat.row(i) = A.row(I(i)); // Select each row in I
    }


    // Compute pseudoinverse
    MatrixXd R_inv = pseudoInverse(R_mat);

    MatrixXd U = Z * R_inv;

    return {C, U, R_mat};
}
