#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace simple_catch {

struct FatalAssertion : std::exception {
  const char* what() const noexcept override { return "fatal assertion"; }
};

struct TestCase {
  const char* name;
  void (*func)();
};

inline std::vector<TestCase>& registry()
{
  static std::vector<TestCase> tests;
  return tests;
}

struct TestRegistrar {
  TestRegistrar(const char* name, void (*func)())
  {
    registry().push_back(TestCase{name, func});
  }
};

inline bool& test_failed()
{
  thread_local bool failed = false;
  return failed;
}

inline bool is_verbose()
{
  static const bool verbose = (std::getenv("VERBOSE") != nullptr);
  return verbose;
}

inline void report_failure(const char* expr, const char* file, int line)
{
  std::cerr << file << ":" << line << ": FAILED: " << expr << std::endl;
  test_failed() = true;
}

} // namespace simple_catch

namespace Catch {

class Approx {
public:
  explicit Approx(double value) noexcept
      : m_value(value)
  {
  }

  Approx& margin(double marginValue) noexcept
  {
    m_margin = marginValue;
    return *this;
  }

  Approx& epsilon(double epsilonValue) noexcept
  {
    m_epsilon = epsilonValue;
    return *this;
  }

  friend bool operator==(double lhs, const Approx& rhs) noexcept
  {
    const double diff = std::fabs(lhs - rhs.m_value);
    const double tol = std::max(rhs.m_margin, rhs.m_epsilon * std::fabs(rhs.m_value));
    return diff <= tol;
  }

  friend bool operator==(const Approx& lhs, double rhs) noexcept
  {
    return rhs == lhs;
  }

  friend bool operator!=(double lhs, const Approx& rhs) noexcept
  {
    return !(lhs == rhs);
  }

  friend bool operator!=(const Approx& lhs, double rhs) noexcept
  {
    return !(lhs == rhs);
  }

private:
  double m_value;
  double m_margin = 0.0;
  double m_epsilon = 1.0e-12;
};

} // namespace Catch

#define CATCH_INTERNAL_CONCAT_IMPL(x, y) x##y
#define CATCH_INTERNAL_CONCAT(x, y) CATCH_INTERNAL_CONCAT_IMPL(x, y)

#define TEST_CASE(name, tags)                                                                        \
  static void CATCH_INTERNAL_CONCAT(test_func_, __LINE__)();                                         \
  static ::simple_catch::TestRegistrar CATCH_INTERNAL_CONCAT(test_reg_, __LINE__)(name,              \
                                                                                   &CATCH_INTERNAL_CONCAT(test_func_, __LINE__)); \
  static void CATCH_INTERNAL_CONCAT(test_func_, __LINE__)()

#define CHECK(expr)                                                                                  \
  do {                                                                                               \
    if (!(expr)) {                                                                                   \
      ::simple_catch::report_failure(#expr, __FILE__, __LINE__);                                     \
    }                                                                                                \
  } while (false)

#define CHECK_FALSE(expr) CHECK(!(expr))

#define REQUIRE(expr)                                                                                \
  do {                                                                                               \
    if (!(expr)) {                                                                                   \
      ::simple_catch::report_failure(#expr, __FILE__, __LINE__);                                     \
      throw ::simple_catch::FatalAssertion();                                                        \
    }                                                                                                \
  } while (false)

#define REQUIRE_FALSE(expr) REQUIRE(!(expr))

namespace simple_catch {

inline int run_all()
{
  const bool verbose = is_verbose();
  int failures = 0;
  for (const TestCase& tc : registry()) {
    if (verbose) {
      std::cout << "[ RUN      ] " << tc.name << std::endl;
    }
    test_failed() = false;
    try {
      tc.func();
    } catch (const FatalAssertion&) {
      // Fatal assertion already recorded via report_failure.
    }

    if (test_failed()) {
      ++failures;
      std::cout << "[  FAILED ] " << tc.name << std::endl;
    } else if (verbose) {
      std::cout << "[       OK ] " << tc.name << std::endl;
    }
  }

  if (failures != 0) {
    std::cout << failures << " test(s) failed" << std::endl;
  } else {
    std::cout << "All tests passed" << std::endl;
  }

  return failures;
}

} // namespace simple_catch

#if !defined(SIMPLE_CATCH_NO_MAIN)
int main()
{
  return ::simple_catch::run_all();
}
#endif
