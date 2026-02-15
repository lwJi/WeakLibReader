#pragma once

#include <AMReX.H>

namespace test_detail {

/// Global AMReX guard that initializes once and finalizes at program exit.
/// This avoids the MPI re-initialization error that occurs when each test
/// creates its own AmrexGuard (MPI cannot be re-initialized after finalization).
struct GlobalAmrexGuard {
  GlobalAmrexGuard()
  {
    int argc = 0;
    char** argv = nullptr;
    amrex::Initialize(argc, argv);
  }
  ~GlobalAmrexGuard() { amrex::Finalize(); }
};

inline GlobalAmrexGuard& GetGlobalAmrexGuard()
{
  static GlobalAmrexGuard guard;
  return guard;
}

/// Per-test helper that ensures AMReX is initialized (via the global guard)
struct AmrexGuard {
  AmrexGuard() { (void)GetGlobalAmrexGuard(); }
};

} // namespace test_detail

using test_detail::AmrexGuard;
