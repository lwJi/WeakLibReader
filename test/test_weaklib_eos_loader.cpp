#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "WeakLibReader_Hdf5Loader.hpp"

#include <AMReX.H>
#include <hdf5.h>

#include <cstdlib>
#include <string>

using namespace WeakLibReader;

namespace {

/// RAII helper that temporarily silences HDF5 automatic error printing.
struct ScopedHdf5ErrorSilencer {
  H5E_auto2_t oldFunc = nullptr;
  void* oldClientData = nullptr;

  ScopedHdf5ErrorSilencer()
  {
    H5Eget_auto2(H5E_DEFAULT, &oldFunc, &oldClientData);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  }

  ~ScopedHdf5ErrorSilencer()
  {
    H5Eset_auto2(H5E_DEFAULT, oldFunc, oldClientData);
  }
};

std::string GetWeakLibTablePath()
{
  const char* env = std::getenv("WEAKLIB_TABLE_PATH");
  if (env != nullptr && env[0] != '\0') {
    return std::string(env);
  }
  // Default path for local development
  return "../tables/wl-EOS-SFHo-15-25-50.h5";
}

/// Initialize AMReX if not already initialized.
/// This relies on AMReX's internal tracking to avoid double-initialization.
void EnsureAmrexInitialized()
{
  if (!amrex::Initialized()) {
    int argc = 0;
    char** argv = nullptr;
    amrex::Initialize(argc, argv);
    // Register finalize at exit - but only if we initialized
    std::atexit([]() {
      if (amrex::Initialized()) {
        amrex::Finalize();
      }
    });
  }
}

} // namespace

TEST_CASE("LoadWeakLibEosTable loads Pressure", "[weaklib][eos]")
{
  EnsureAmrexInitialized();

  const std::string path = GetWeakLibTablePath();
  Hdf5Table table;
  const Hdf5LoadStatus status = LoadWeakLibEosTable(path, "Pressure", table);

  REQUIRE(status == Hdf5LoadStatus::Success);

  // Verify dimensions (C order: rho, T, Ye)
  REQUIRE(table.nd == 3);
  CHECK(table.extents[0] == 185);  // Density
  CHECK(table.extents[1] == 81);   // Temperature
  CHECK(table.extents[2] == 30);   // Electron Fraction

  // Verify axis scales
  CHECK(table.axes[0].scale == AxisScale::Log10);   // Density
  CHECK(table.axes[1].scale == AxisScale::Log10);   // Temperature
  CHECK(table.axes[2].scale == AxisScale::Linear);  // Ye

  // Verify axis grid sizes
  CHECK(table.axes[0].n == 185);
  CHECK(table.axes[1].n == 81);
  CHECK(table.axes[2].n == 30);

  // Verify axis grids are valid pointers
  REQUIRE(table.axes[0].grid != nullptr);
  REQUIRE(table.axes[1].grid != nullptr);
  REQUIRE(table.axes[2].grid != nullptr);

  // Verify data pointer is valid
  REQUIRE(table.DataPtr() != nullptr);

  // Verify layout
  CHECK(table.layout.nd == 3);
  CHECK(table.layout.n[0] == 185);
  CHECK(table.layout.n[1] == 81);
  CHECK(table.layout.n[2] == 30);
}

TEST_CASE("LoadWeakLibEosTable loads Entropy Per Baryon", "[weaklib][eos]")
{
  EnsureAmrexInitialized();

  const std::string path = GetWeakLibTablePath();
  Hdf5Table table;
  const Hdf5LoadStatus status = LoadWeakLibEosTable(path, "Entropy Per Baryon", table);

  REQUIRE(status == Hdf5LoadStatus::Success);
  CHECK(table.nd == 3);
  CHECK(table.extents[0] == 185);
  CHECK(table.extents[1] == 81);
  CHECK(table.extents[2] == 30);
}

TEST_CASE("LoadWeakLibEosTable fails for nonexistent variable", "[weaklib][eos]")
{
  EnsureAmrexInitialized();
  ScopedHdf5ErrorSilencer silencer{};

  const std::string path = GetWeakLibTablePath();
  Hdf5Table table;
  const Hdf5LoadStatus status = LoadWeakLibEosTable(path, "NonexistentVariable", table);

  CHECK(status == Hdf5LoadStatus::DatasetOpenFailed);
}

TEST_CASE("LoadWeakLibEosTable fails for nonexistent file", "[weaklib][eos]")
{
  EnsureAmrexInitialized();
  ScopedHdf5ErrorSilencer silencer{};

  Hdf5Table table;
  const Hdf5LoadStatus status = LoadWeakLibEosTable("/nonexistent/path.h5", "Pressure", table);

  CHECK(status == Hdf5LoadStatus::FileOpenFailed);
}

TEST_CASE("LoadWeakLibEosTable axis values are reasonable", "[weaklib][eos]")
{
  EnsureAmrexInitialized();

  const std::string path = GetWeakLibTablePath();
  Hdf5Table table;
  const Hdf5LoadStatus status = LoadWeakLibEosTable(path, "Pressure", table);

  REQUIRE(status == Hdf5LoadStatus::Success);

  // Density axis: should be positive (for log10), roughly 1e3 to 1e15 g/cm^3
  const double* rho = table.axes[0].grid;
  CHECK(rho[0] > 0.0);
  CHECK(rho[0] < rho[table.axes[0].n - 1]);  // Monotonically increasing

  // Temperature axis: should be positive, roughly 0.1 to 100 MeV
  const double* T = table.axes[1].grid;
  CHECK(T[0] > 0.0);
  CHECK(T[0] < T[table.axes[1].n - 1]);

  // Electron fraction: 0 to 0.6 typically
  const double* Ye = table.axes[2].grid;
  CHECK(Ye[0] >= 0.0);
  CHECK(Ye[0] < Ye[table.axes[2].n - 1]);
  CHECK(Ye[table.axes[2].n - 1] <= 1.0);
}

TEST_CASE("LoadWeakLibEosTable data values spot check", "[weaklib][eos][values]")
{
  EnsureAmrexInitialized();

  const std::string path = GetWeakLibTablePath();
  Hdf5Table table;
  const Hdf5LoadStatus status = LoadWeakLibEosTable(path, "Pressure", table);

  REQUIRE(status == Hdf5LoadStatus::Success);

  // Access data using layout
  const double* data = table.DataPtr();
  const Layout& layout = table.layout;

  // Check corner values (indices in C order: rho, T, Ye)
  // Value at (0, 0, 0) - first element
  const std::size_t idx000 = layout.Offset(0, 0, 0);
  CHECK(data[idx000] > 0.0);  // Pressure should be positive

  // Value at last index
  const int lastRho = table.extents[0] - 1;
  const int lastT = table.extents[1] - 1;
  const int lastYe = table.extents[2] - 1;
  const std::size_t idxLast = layout.Offset(lastRho, lastT, lastYe);
  CHECK(data[idxLast] > 0.0);

  // Total size check
  const std::size_t expectedSize = 185UL * 81UL * 30UL;
  CHECK(layout.Offset(lastRho, lastT, lastYe) == expectedSize - 1);
}
