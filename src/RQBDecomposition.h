// RQBDecomposition.h
#ifndef RQB_DECOMPOSITION_H
#define RQB_DECOMPOSITION_H

#include <Eigen/Dense>
#include <string>

struct RQBDecomposition {
    Eigen::MatrixXd Q;
    Eigen::MatrixXd B;
};

RQBDecomposition rqb(const Eigen::MatrixXd& A, int k, int p = 10, int q = 0,
                     const std::string& dist = "normal", bool rand = true);

#endif
