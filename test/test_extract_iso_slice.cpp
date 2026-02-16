#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "hdf5/WeakLibReader_Hdf5LoaderOpacityScat.hpp"
#include "base/WeakLibReader_Layout.hpp"

#include <vector>

TEST_CASE("ExtractIsoMomentSlice4D extracts correct 4D slice", "[isoslice]")
{
  using namespace WeakLibReader;

  // 5D dimensions: [nE=3, nMom=2, nRho=4, nT=3, nYe=2]
  const std::array<int, 5> dims{{3, 2, 4, 3, 2}};
  const int nE   = dims[0];
  const int nMom = dims[1];
  const int nRho = dims[2];
  const int nT   = dims[3];
  const int nYe  = dims[4];

  const Layout layout5d = MakeLayout(dims.data(), 5);
  const std::size_t totalSize =
      static_cast<std::size_t>(nE) * nMom * nRho * nT * nYe;

  // Fill 5D array with index-encoding:
  // data[offset] = iE + 10*iMom + 100*iRho + 1000*iT + 10000*iYe
  std::vector<double> data(totalSize);
  for (int iYe = 0; iYe < nYe; ++iYe) {
    for (int iT = 0; iT < nT; ++iT) {
      for (int iRho = 0; iRho < nRho; ++iRho) {
        for (int iMom = 0; iMom < nMom; ++iMom) {
          for (int iE = 0; iE < nE; ++iE) {
            const double val = iE + 10 * iMom + 100 * iRho
                             + 1000 * iT + 10000 * iYe;
            data[layout5d.Offset(iE, iMom, iRho, iT, iYe)] = val;
          }
        }
      }
    }
  }

  // 4D layout for verification: [nE, nRho, nT, nYe]
  const std::array<int, 4> dims4d{{nE, nRho, nT, nYe}};
  const Layout layout4d = MakeLayout(dims4d.data(), 4);

  for (int mom = 0; mom < nMom; ++mom) {
    const auto result = ExtractIsoMomentSlice4D(data.data(), dims, mom);

    REQUIRE(result.size() ==
            static_cast<std::size_t>(nE) * nRho * nT * nYe);

    for (int iYe = 0; iYe < nYe; ++iYe) {
      for (int iT = 0; iT < nT; ++iT) {
        for (int iRho = 0; iRho < nRho; ++iRho) {
          for (int iE = 0; iE < nE; ++iE) {
            const double expected = iE + 10 * mom + 100 * iRho
                                  + 1000 * iT + 10000 * iYe;
            CHECK(result[layout4d.Offset(iE, iRho, iT, iYe)]
                  == expected);
          }
        }
      }
    }
  }
}

TEST_CASE("ExtractIsoMomentSlice4D single-moment degenerate case",
          "[isoslice]")
{
  using namespace WeakLibReader;

  // 5D dimensions: [nE=2, nMom=1, nRho=2, nT=2, nYe=2]
  const std::array<int, 5> dims{{2, 1, 2, 2, 2}};
  const int nE   = dims[0];
  const int nMom = dims[1];
  const int nRho = dims[2];
  const int nT   = dims[3];
  const int nYe  = dims[4];

  const Layout layout5d = MakeLayout(dims.data(), 5);
  const std::size_t totalSize =
      static_cast<std::size_t>(nE) * nMom * nRho * nT * nYe;

  // Fill with index-encoding (nMom=1, so iMom always 0)
  std::vector<double> data(totalSize);
  for (int iYe = 0; iYe < nYe; ++iYe) {
    for (int iT = 0; iT < nT; ++iT) {
      for (int iRho = 0; iRho < nRho; ++iRho) {
        for (int iE = 0; iE < nE; ++iE) {
          const double val = iE + 100 * iRho + 1000 * iT + 10000 * iYe;
          data[layout5d.Offset(iE, 0, iRho, iT, iYe)] = val;
        }
      }
    }
  }

  const auto result = ExtractIsoMomentSlice4D(data.data(), dims, 0);

  // With nMom=1 and iMom=0, the 5D data collapses to 4D.
  // Since stride[0]=1 and the moment dimension has size 1,
  // 5D strides [1, nE, nE*1, ...] become effectively [1, nE, nE*nRho, ...]
  // which matches 4D strides [1, nE, nE*nRho, ...].
  // So output should match input element-for-element.
  REQUIRE(result.size() ==
          static_cast<std::size_t>(nE) * nRho * nT * nYe);

  const std::array<int, 4> dims4d{{nE, nRho, nT, nYe}};
  const Layout layout4d = MakeLayout(dims4d.data(), 4);

  for (int iYe = 0; iYe < nYe; ++iYe) {
    for (int iT = 0; iT < nT; ++iT) {
      for (int iRho = 0; iRho < nRho; ++iRho) {
        for (int iE = 0; iE < nE; ++iE) {
          const double expected = iE + 100 * iRho + 1000 * iT + 10000 * iYe;
          CHECK(result[layout4d.Offset(iE, iRho, iT, iYe)] == expected);
        }
      }
    }
  }
}
