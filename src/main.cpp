#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <stdexcept>
#include <Eigen/QR>
#include <vector>

#include "RCUR_4_debug.h"


int main() {
    // Example matrix A
    MatrixXd A = MatrixXd::Random(10, 8);
    int k = 3, p = 2, q = 2;
    bool rand;

    cout << "Select the algorithm to use (0 = Deterministic, 1 = Randomized): " << endl;
    cin >> rand;
    cout << "A:\n" << A << "\n";

    try {
        // Perform Randomized CUR
        auto [C, U, R] = rcur(A, k, p, q, rand);

        // Output results
        cout << "C:\n" << C << "\n";
        cout << "U:\n" << U << "\n";
        cout << "R:\n" << R << "\n";

        cout << "CUR:\n" << C * U * R << "\n";

    } catch (const invalid_argument &e) {
        cerr << "Error: " << e.what() << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}