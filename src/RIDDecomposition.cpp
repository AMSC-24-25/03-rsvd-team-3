#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "RQBDecomposition.h" // Include per il metodo RQB


struct RIDDecomposition {
    Eigen::MatrixXd C;
    Eigen::MatrixXd R;
    Eigen::MatrixXd Z;
    std::vector<int> indices;
    std::vector<int> pivot;
    std::vector<double> scores;
    std::vector<double> scores_idx;
    std::string mode;
    bool rand;
};

RIDDecomposition rid(const Eigen::MatrixXd& A_input, int k, const std::string& mode = "column", int p = 10, int q = 0, bool idx_only = false, bool rand = true) {
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Inizializzazione variabili
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Eigen::MatrixXd A = A_input;
    int m = A.rows();
    int n = A.cols();

    if (mode == "row") {
        A.transposeInPlace();
        std::swap(m, n);
    }

    if (k > std::min(m, n) || k < 1) {
        throw std::invalid_argument("Target rank is not valid!");
    }

    RIDDecomposition idObj;
    idObj.mode = mode;
    idObj.rand = rand;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Calcolo della decomposizione interpolativa
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr;
    if (rand) {
        RQBDecomposition rqbRes = rqb(A, k, p, q, "normal", true);
        qr.compute(rqbRes.B);
    } else {
        qr.compute(A);
    }

    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
    Eigen::VectorXi pivot = qr.colsPermutation().indices();

    idObj.indices.resize(k);
    for (int i = 0; i < k; ++i) {
        idObj.indices[i] = pivot(i);
    }

    idObj.pivot.resize(pivot.size());
    for (int i = 0; i < pivot.size(); ++i) {
        idObj.pivot[i] = pivot(i);
    }

    // Ordina i pivot
    std::vector<int> ordered_pivot(pivot.data(), pivot.data() + pivot.size());
    std::sort(ordered_pivot.begin(), ordered_pivot.end());

    // Calcola i punteggi
    idObj.scores.resize(k);
    idObj.scores_idx.resize(k);
    for (int i = 0; i < k; ++i) {
        idObj.scores[i] = std::abs(R(i, i));
        idObj.scores_idx[i] = idObj.scores[i];
    }

    idObj.scores = std::vector<double>(idObj.scores.begin(), idObj.scores.end());

    // Calcolo di Z
    if (k == n) {
        idObj.Z = R.topLeftCorner(k, k).inverse();
    } else {
        Eigen::MatrixXd V = R.topLeftCorner(k, k).inverse() * R.topRightCorner(k, n - k);
        idObj.Z = Eigen::MatrixXd::Identity(k, n);
        idObj.Z.block(0, k, k, n - k) = V;
    }

    Eigen::MatrixXd Z_reordered(k, n);
    for (int i = 0; i < n; ++i) {
        Z_reordered.col(i) = idObj.Z.col(ordered_pivot[i]);
    }
    idObj.Z = Z_reordered;

    // Ripristina la trasposizione se il mode è "row"
    if (mode == "row") {
        idObj.Z.transposeInPlace();
    }

    // Selezione delle colonne o righe
    if (!idx_only) {
        if (mode == "column") {
            idObj.C.resize(m, k);
            for (int i = 0; i < k; ++i) {
                idObj.C.col(i) = A.col(idObj.indices[i]);
            }
        } else {
            idObj.R.resize(k, n);
            for (int i = 0; i < k; ++i) {
                idObj.R.row(i) = A.row(idObj.indices[i]);
            }
        }
    }

    return idObj;
}
