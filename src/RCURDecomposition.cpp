#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "RIDDecomposition.h" // Include necessario per il rid()
#include "RQBDecomposition.h"

struct CURDecomposition {
    Eigen::MatrixXd C;
    Eigen::MatrixXd U;
    Eigen::MatrixXd R;
    std::vector<int> C_idx;
    std::vector<int> R_idx;
    std::vector<double> C_scores;
    std::vector<double> R_scores;
    bool rand;
};

// Funzione per calcolare la pseudo-inversa
Eigen::MatrixXd pinv(const Eigen::MatrixXd& mat, double tolerance = 1e-9) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tol = tolerance * std::max(mat.rows(), mat.cols()) * svd.singularValues().array().abs().maxCoeff();
    
    Eigen::VectorXd singularValuesInv(svd.singularValues().size());
    for (int i = 0; i < svd.singularValues().size(); ++i) {
        if (svd.singularValues()(i) > tol) {
            singularValuesInv(i) = 1.0 / svd.singularValues()(i);
        } else {
            singularValuesInv(i) = 0.0;
        }
    }

    return svd.matrixV() * singularValuesInv.asDiagonal() * svd.matrixU().transpose();
}

CURDecomposition rcur(const Eigen::MatrixXd& A, int k, int p = 10, int q = 0, bool idx_only = false, bool rand = true) {
    int m = A.rows();
    int n = A.cols();

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Verifica e settaggio del rango target
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if (k <= 0 || k > std::min(m, n)) {
        throw std::invalid_argument("Target rank is not valid!");
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Compute column ID
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RIDDecomposition colRID = rid(A, k, "column", p, q, true, rand);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Select column subset
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Eigen::MatrixXd C;
    if (!idx_only) {
        C.resize(m, k);
        for (int i = 0; i < k; ++i) {
            C.col(i) = A.col(colRID.indices[i]);
        }
    }

    std::vector<int> C_idx = colRID.indices;
    std::vector<double> C_scores(colRID.scores.begin(), colRID.scores.end());

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Compute row ID of C
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Eigen::MatrixXd A_C(m, k);
    for (int i = 0; i < k; ++i) {
        A_C.col(i) = A.col(colRID.indices[i]);
    }

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A_C.transpose());
    Eigen::VectorXi pivots = qr.colsPermutation().indices();

    std::vector<int> R_idx(k);
    for (int i = 0; i < k; ++i) {
        R_idx[i] = pivots(i);
    }

    Eigen::MatrixXd S = qr.matrixQR().triangularView<Eigen::Upper>();

    Eigen::MatrixXd R;
    if (!idx_only) {
        R.resize(k, n);
        for (int i = 0; i < k; ++i) {
            R.row(i) = A.row(R_idx[i]);
        }
    }

    std::vector<double> R_scores(k);
    for (int i = 0; i < k; ++i) {
        R_scores[i] = std::abs(S(i, i));
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Compute U
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Eigen::MatrixXd R_subset(k, n);
    for (int i = 0; i < k; ++i) {
        R_subset.row(i) = A.row(R_idx[i]);
    }

    Eigen::MatrixXd U = colRID.Z * pinv(R_subset);

    // //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // // Alternative U computation
    // //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // // Create a diagonal matrix diag(k)
    // Eigen::MatrixXd diag_k = Eigen::MatrixXd::Identity(k, k);

    // // Compute V = [diag(k); H(out_cid.Z)]
    // Eigen::MatrixXd V;
    // V.resize(2 * k, k); // Assuming 2k for the vertical stack
    // V << diag_k, colRID.Z.transpose();

    // // Rt * Ut = V via RRt * Ut = R * V
    // Eigen::MatrixXd RV = R * V;
    // U = pinv(R * R.transpose()) * RV;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Return CURDecomposition object
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return {C, U, R, C_idx, R_idx, C_scores, R_scores, rand};
}
