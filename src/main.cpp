#include "RandomizedSVD.h"
#include <iostream>


int main() {
    Eigen::MatrixXd A(3, 3);
    A << 1, 3, 2,
         5, 3, 1,
         3, 4, 5;

    Eigen::RandomizedSVD<Eigen::MatrixXd> rsvd;

    constexpr int rank {2};
    constexpr int powerIter {3};

    rsvd.compute(A, rank, powerIter);

    const auto& singularValues = rsvd.singularValues();
    const auto& U = rsvd.matrixU();
    const auto& V = rsvd.matrixV();

    std::cout << "Singular values: " << singularValues.transpose() << std::endl;
    std::cout << "U matrix: " << U << std::endl;
    std::cout << "V matrix: " << V << std::endl;

    return 0;
}