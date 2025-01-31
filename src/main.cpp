#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include "RCUR_4_debug.h"

using namespace Eigen;
using namespace std;

void testMatrix(int rows, int cols, int rank, int p, int q, bool rand, double epsilon) {
    cout << "\n===== Testing Matrix: " << rows << "x" << cols << " with rank " << rank 
         << " | " << (rand ? "Randomized" : "Deterministic") << " CUR =====" << endl;

    // Genera una matrice di basso rango
    MatrixXd A = MatrixXd::Random(rows, rank) * MatrixXd::Random(rank, cols);
    cout << "Generated matrix A of size " << rows << "x" << cols << " and effective rank " << rank << endl;

    try {
        // Calcolo della decomposizione CUR
        auto [C, U, R] = rcur(A, rank, p, q, rand);
        MatrixXd CUR = C * U * R;

        // Calcolo della decomposizione SVD troncata
        JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
        MatrixXd U_k = svd.matrixU().leftCols(rank);
        MatrixXd S_k = svd.singularValues().head(rank).asDiagonal();
        MatrixXd V_k = svd.matrixV().leftCols(rank);
        MatrixXd A_k = U_k * S_k * V_k.transpose();

        // Calcolo dell'errore relativo in norma di Frobenius
        double frob_A = A.norm();
        double error_CUR = (A - CUR).norm() / frob_A;
        double error_SVD = (A - A_k).norm() / frob_A;

        // Definizione della soglia (2+ε) * errore_SVD
        double threshold = (2 + epsilon) * error_SVD;

        cout << "Relative Frobenius error for CUR: " << error_CUR << endl;
        cout << "Relative Frobenius error for SVD: " << error_SVD << endl;
        cout << "Threshold (2+ε) * error_SVD: " << threshold << endl;

        // Verifica se CUR ha errore minore del bound (2+ε) * errore_SVD
        if (error_CUR < threshold) {
            cout << "CUR satisfies the (2+ε) error bound." << endl;
        } else {
            cout << "SVD approximation is significantly better than CUR." << endl;
        }

    } catch (const invalid_argument &e) {
        cerr << "Error: " << e.what() << endl;
    }
}

int main() {
    int p = 2, q = 2;
    double epsilon = 1e-6;

    // CASI DI TEST: Randomized e Deterministico
    vector<pair<int, int>> matrix_sizes = {{500, 400}, {500, 400}, {10, 8}, {10, 8}};
    vector<int> ranks = {20, 350, 3, 7};

    for (size_t i = 0; i < matrix_sizes.size(); ++i) {
        int rows = matrix_sizes[i].first;
        int cols = matrix_sizes[i].second;
        int rank = ranks[i];

        // Test Randomized CUR
        testMatrix(rows, cols, rank, p, q, true, epsilon);
        
        // Test Deterministic CUR
        testMatrix(rows, cols, rank, p, q, false, epsilon);
    }

    return 0;
}
