#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include "RQBDecomposition.h"
#include <complex>

// Funzione per generare una matrice casuale
Eigen::MatrixXd generateRandomMatrix(int rows, int cols, const std::string& dist = "normal") {
    Eigen::MatrixXd randomMatrix(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if (dist == "normal") {
        std::normal_distribution<> d(0, 1);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                randomMatrix(i, j) = d(gen);
    } else if (dist == "unif") {
        std::uniform_real_distribution<> d(-1, 1);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                randomMatrix(i, j) = d(gen);
    } else if (dist == "rademacher") {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                randomMatrix(i, j) = (rand() % 2 == 0) ? 1 : -1;
    } else {
        throw std::invalid_argument("Unsupported distribution type.");
    }

    return randomMatrix;
}

// Funzione RQB
struct RQBDecomposition {
    Eigen::MatrixXd Q;
    Eigen::MatrixXd B;
};

RQBDecomposition rqb(const Eigen::MatrixXd& A, int k, int p = 10, int q = 2, const std::string& dist = "normal", bool rand = true) {
    int m = A.rows();
    int n = A.cols();

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Verifica e settaggio del rango target
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if (k <= 0 || k > n) {
        throw std::invalid_argument("Target rank is not valid!");
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Imposta il parametro di sovracampionamento
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int l = k + p;
    if (l > n) l = n;
    if (l < 1) {
        throw std::invalid_argument("Oversampling parameter is not valid!");
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Verifica se la matrice è reale o complessa
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    bool isReal = (A.imag().norm() == 0);

    Eigen::MatrixXd Q, B;

    if (rand) {
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // Genera una matrice di campionamento casuale O
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Eigen::MatrixXd O = generateRandomMatrix(n, l, dist);

        if (!isReal) {
            O += std::complex<double>(0, 1) * generateRandomMatrix(n, l, dist);
        }

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // Costruisci la matrice di campionamento Y : Y = A * O
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Eigen::MatrixXd Y = A * O;

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // Iterazioni di potenza
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (q > 0) {
            for (int i = 0; i < q; ++i) {
                Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(Y);
                Eigen::MatrixXd Z = A.transpose() * qr.householderQ();
                Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qrZ(Z);
                Y = A * qrZ.householderQ();
            }
        }

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // Ortogonalizza Y utilizzando la decomposizione QR
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qrY(Y);
        Q = qrY.householderQ();
    } else {
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // Decomposizione deterministica QR di A
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qrA(A);
        Q = qrA.householderQ();
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Proietta A nello spazio di Q per ottenere B
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    B = Q.adjoint() * A;

    return {Q.leftCols(k), B.topRows(k)};
}
