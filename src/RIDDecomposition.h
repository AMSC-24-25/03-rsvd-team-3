// RIDDecomposition.h
#ifndef RID_DECOMPOSITION_H
#define RID_DECOMPOSITION_H

#include <Eigen/Dense>
#include <vector>

struct RIDDecomposition {
    Eigen::MatrixXd Z;
    std::vector<int> indices;
    std::vector<int> pivot;
    std::vector<double> scores;
    std::vector<double> scores_idx;
    Eigen::MatrixXd C;
    Eigen::MatrixXd R;
};

RIDDecomposition rid(const Eigen::MatrixXd& A_input, int k, const std::string& mode = "column",
                     int p = 10, int q = 0, bool idx_only = false, bool rand = true);

#endif
