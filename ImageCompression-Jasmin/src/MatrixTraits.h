#ifndef MATRIXTRAITS_H
#define MATRIXTRAITS_H

#include <Eigen/SparseCore>
#include <type_traits>

namespace Eigen {

    /**
     * @brief Primary template: by default, assume the matrix is dense
     */
    template<typename T>
    struct is_sparse_matrix : std::false_type {};

    /**
     * @brief Specialization for SparseMatrix
     */
    template<typename Scalar_, int Options_, typename Index_>
    struct is_sparse_matrix<SparseMatrix<Scalar_, Options_, Index_>> : std::true_type {};

} // namespace Eigen

#endif // MATRIXTRAITS_H