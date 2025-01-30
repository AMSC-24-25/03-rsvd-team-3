#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "CURDecomposition.h" // Includi la funzione rcur()

// Test: CUR per matrici reali
TEST(CURDecompositionTest, RealMatrix) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(10, 8); // Matrice casuale 10x8
    int k = 5;

    CURDecomposition result = rcur(A, k);

    // Verifica dimensioni
    EXPECT_EQ(result.C.rows(), 10); // 10 righe
    EXPECT_EQ(result.C.cols(), k); // k colonne
    EXPECT_EQ(result.R.rows(), k); // k righe
    EXPECT_EQ(result.R.cols(), 8); // 8 colonne
    EXPECT_EQ(result.U.rows(), k); // k righe
    EXPECT_EQ(result.U.cols(), k); // k colonne

    // Verifica la ricostruzione della matrice
    Eigen::MatrixXd reconstructed = result.C * result.U * result.R;
    ASSERT_TRUE((A - reconstructed).norm() < 1e-10) << "La matrice ricostruita non è accurata.";
}

// Test: CUR con oversampling
TEST(CURDecompositionTest, Oversampling) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(15, 10); // Matrice casuale 15x10
    int k = 5;
    int p = 5; // Oversampling

    CURDecomposition result = rcur(A, k, p);

    // Verifica dimensioni
    EXPECT_EQ(result.C.rows(), 15); // 15 righe
    EXPECT_EQ(result.C.cols(), k); // k colonne
    EXPECT_EQ(result.R.rows(), k); // k righe
    EXPECT_EQ(result.R.cols(), 10); // 10 colonne
    EXPECT_EQ(result.U.rows(), k); // k righe
    EXPECT_EQ(result.U.cols(), k); // k colonne

    // Verifica la ricostruzione della matrice
    Eigen::MatrixXd reconstructed = result.C * result.U * result.R;
    ASSERT_TRUE((A - reconstructed).norm() < 1e-10) << "La matrice ricostruita non è accurata.";
}
