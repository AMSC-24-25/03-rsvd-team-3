#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "RIDDecomposition.h" // Includi la funzione rid()

// Test: RID per matrici reali in modalità column
TEST(RIDDecompositionTest, RealMatrixColumnMode) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(10, 8); // Matrice casuale 10x8
    int k = 5;

    RIDDecomposition result = rid(A, k, "column");

    // Verifica dimensioni
    EXPECT_EQ(result.C.rows(), 10); // 10 righe
    EXPECT_EQ(result.C.cols(), k); // k colonne
    EXPECT_EQ(result.Z.rows(), k); // k righe
    EXPECT_EQ(result.Z.cols(), 8); // 8 colonne

    // Verifica la ricostruzione della matrice
    Eigen::MatrixXd reconstructed = result.C * result.Z;
    ASSERT_TRUE((A - reconstructed).norm() < 1e-10) << "La matrice ricostruita non è accurata.";
}

// Test: RID per matrici reali in modalità row
TEST(RIDDecompositionTest, RealMatrixRowMode) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(8, 10); // Matrice casuale 8x10
    int k = 5;

    RIDDecomposition result = rid(A, k, "row");

    // Verifica dimensioni
    EXPECT_EQ(result.R.rows(), k); // k righe
    EXPECT_EQ(result.R.cols(), 10); // 10 colonne
    EXPECT_EQ(result.Z.rows(), 8); // 8 righe
    EXPECT_EQ(result.Z.cols(), k); // k colonne

    // Verifica la ricostruzione della matrice
    Eigen::MatrixXd reconstructed = result.Z * result.R;
    ASSERT_TRUE((A - reconstructed).norm() < 1e-10) << "La matrice ricostruita non è accurata.";
}
