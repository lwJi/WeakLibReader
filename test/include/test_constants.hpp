#pragma once

namespace test_constants {

// Default tolerance for numerical comparisons across the test suite.
// Individual test files may define a tighter or looser tolerance locally.
constexpr double Tol = 1.0e-12;

} // namespace test_constants
