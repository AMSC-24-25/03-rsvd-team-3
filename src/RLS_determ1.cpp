#include <iostream>
#include <vector>
#include <algorithm> // Per sort e max_element
#include <numeric>   // Per std::iota
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Funzione per calcolare CUR decomposition in modo deterministico
MatrixXd CURDecompositionDeterministic(const MatrixXd& A, int k) {
    // Step 1: Compute SVD
    JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    MatrixXd U = svd.matrixU();    // Left singular vectors
    MatrixXd V = svd.matrixV();    // Right singular vectors
    // VectorXd S = svd.singularValues(); // Singular values

    // Step 2: Compute leverage scores for columns
    VectorXd leverage_scores_cols = V.leftCols(k).rowwise().squaredNorm() / k;

    // Step 3: Select top-k columns based on leverage scores
    vector<int> selected_columns(k);
    iota(selected_columns.begin(), selected_columns.end(), 0);
    partial_sort(selected_columns.begin(), selected_columns.begin() + k, selected_columns.end(),
                 [&leverage_scores_cols](int i, int j) { return leverage_scores_cols[i] > leverage_scores_cols[j]; });

    // Construct matrix C
    MatrixXd C(A.rows(), k);
    for (int i = 0; i < k; ++i) {
        C.col(i) = A.col(selected_columns[i]);
    }

    cout << "Matrice C:\n" << C << endl << endl;

    // Step 4: Compute leverage scores for rows
    VectorXd leverage_scores_rows = U.leftCols(k).rowwise().squaredNorm() / k;

    // Step 5: Select top-k rows based on leverage scores
    vector<int> selected_rows(k);
    iota(selected_rows.begin(), selected_rows.end(), 0);
    partial_sort(selected_rows.begin(), selected_rows.begin() + k, selected_rows.end(),
                 [&leverage_scores_rows](int i, int j) { return leverage_scores_rows[i] > leverage_scores_rows[j]; });

    // Construct matrix R
    MatrixXd R(k, A.cols());
    for (int i = 0; i < k; ++i) {
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

    cout << "Matrice A:\n" << A << endl << endl;
    
    // Compute CUR decomposition deterministically
    MatrixXd CUR = CURDecompositionDeterministic(A, k);
    cout << "Deterministic CUR Decomposition completed!" << endl;
    cout << "Approximated matrix (CUR):" << endl << CUR << endl;

    return 0;
}
