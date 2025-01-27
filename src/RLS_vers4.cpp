#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Funzione per calcolare CUR decomposition
MatrixXd CURDecomposition(const MatrixXd& A, int k, double epsilon) {
    // Step 1: Compute SVD
    JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    MatrixXd U = svd.matrixU();    // Left singular vectors
    MatrixXd V = svd.matrixV();    // Right singular vectors
    // VectorXd S = svd.singularValues(); // Singular values

    // Step 2: Compute leverage scores for columns
    VectorXd leverage_scores_cols = V.leftCols(k).rowwise().squaredNorm() / k;

    // Step 3: Sample columns based on leverage scores
    vector<int> selected_columns;
    default_random_engine generator;
    for (int i = 0; i < A.cols(); ++i) {
        uniform_real_distribution<double> distribution(0.0, 1.0);
        if (distribution(generator) < leverage_scores_cols[i]) {
            selected_columns.push_back(i);
        }
    }

    // Construct matrix C
    MatrixXd C(A.rows(), selected_columns.size());
    for (size_t i = 0; i < selected_columns.size(); ++i) {
        C.col(i) = A.col(selected_columns[i]);
    }
    
    cout << "Matrice C:\n" << C << endl << endl;

    // Step 4: Compute leverage scores for rows
    VectorXd leverage_scores_rows = U.leftCols(k).rowwise().squaredNorm() / k;

    // Step 5: Sample rows based on leverage scores
    vector<int> selected_rows;
    for (int i = 0; i < A.rows(); ++i) {
        uniform_real_distribution<double> distribution(0.0, 1.0);
        if (distribution(generator) < leverage_scores_rows[i]) {
            selected_rows.push_back(i);
        }
    }

    // Construct matrix R
    MatrixXd R(selected_rows.size(), A.cols());
    for (size_t i = 0; i < selected_rows.size(); ++i) {
        R.row(i) = A.row(selected_rows[i]);
    }

    cout << "Matrice R:\n" << R << endl << endl;

    // Step 6: Compute U = C^+ A R^+
    MatrixXd C_pseudo = C.completeOrthogonalDecomposition().pseudoInverse();
    MatrixXd R_pseudo = R.completeOrthogonalDecomposition().pseudoInverse();
    MatrixXd UU = C_pseudo * A * R_pseudo;

    cout << "Matrice U:\n" << UU << endl << endl;
    
    return C * UU * R; // Return the CUR decomposition
}

int main() {
    // Example input matrix
    MatrixXd A = MatrixXd::Random(6, 4); // Random 100x50 matrix
    int k = 2; // Target rank
    double epsilon = 0.1; // Error tolerance
    
    cout << "Matrice A:\n" << A << endl << endl;

    // Compute CUR decomposition
    MatrixXd CUR = CURDecomposition(A, k, epsilon);
    cout << "CUR Decomposition completed!" << endl;
    cout << "Approximated matrix (CUR):" << endl << CUR << endl;

    return 0;
}
