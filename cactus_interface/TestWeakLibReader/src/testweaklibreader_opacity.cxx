#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <loop_device.hxx>

#include <vector>

#include <hdf5/WeakLibReader_Hdf5Loader.hpp>
#include <interp/WeakLibReader_LogInterpolate.hpp>

#include "testweaklibreader_helpers.hxx"

namespace TestWeakLibReader {

WeakLibReader::WeakLibOpacityTableDevice opacity_table_device;

// Fixed energy value at midpoint of energy grid (set during load)
double fixedEnergyMidpoint = 0.0;

// Iso moment=0 slice, species=0, copied to device
amrex::Gpu::DeviceVector<double> iso_slice_device;

// Pre-aligned scattering kernel tables (host-computed, device-stored)
struct AlignedKernel {
  amrex::Gpu::DeviceVector<double> data;
  int nAlignedE = 0;
  int nDim3 = 0;   // nT for NES/Pair, nRho for Brem
  int nDim4 = 0;   // nEta for NES/Pair, nT for Brem
  WeakLibReader::Layout layout{};
};

std::vector<AlignedKernel> nes_aligned;
std::vector<AlignedKernel> pair_aligned;
std::vector<AlignedKernel> brem_aligned;

extern "C" void TestWeakLibReader_LoadOpacityTable(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  const std::string emabFile(opacity_emab_file);
  const std::string isoFile(opacity_iso_file);
  const std::string nesFile(opacity_nes_file);
  const std::string pairFile(opacity_pair_file);
  const std::string bremFile(opacity_brem_file);

  if (emabFile.empty() && isoFile.empty() && nesFile.empty() &&
      pairFile.empty() && bremFile.empty()) {
    CCTK_INFO("No opacity table files specified, skipping opacity load");
    return;
  }

  CCTK_INFO("Loading opacity tables");

  WeakLibReader::WeakLibOpacityTable opacity_table;
  const auto status = WeakLibReader::LoadWeakLibOpacityTableFullParallel(
      opacity_table, emabFile, isoFile, nesFile, pairFile, bremFile);

  if (status != WeakLibReader::Hdf5LoadStatus::Success) {
    CCTK_ERROR("Failed to load opacity tables");
  }

  // Store midpoint energy from host-side table before device copy
  fixedEnergyMidpoint =
      opacity_table.energyGrid.values[opacity_table.energyGrid.nPoints / 2];

  opacity_table_device = WeakLibReader::MakeDeviceCopy(opacity_table);

  // Extract Iso moment=0 slice as contiguous 4D and copy to device
  if (opacity_table.scatIso.IsLoaded()) {
    auto slice = WeakLibReader::ExtractIsoMomentSlice4D(
        opacity_table.scatIso.KernelData(0),
        opacity_table.scatIso.dimensions, /*iMom=*/0);
    iso_slice_device.resize(slice.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     slice.begin(), slice.end(),
                     iso_slice_device.begin());
    CCTK_VINFO("Iso slice extracted: %zu elements", slice.size());
  }

  // Pre-align NES
  if (opacity_table.scatNES.IsLoaded()) {
    const auto& nes = opacity_table.scatNES;
    const int nE = opacity_table.energyGrid.nPoints;
    const int nMom = nes.nMoments;
    const int nT = nes.dimensions[3];
    const int nEta = nes.dimensions[4];
    const auto energyAxis = opacity_table.energyGrid.MakeAxis();
    const auto nesLayout = WeakLibReader::MakeLayout(nes.dimensions.data(), 5);

    nes_aligned.resize(nMom);
    for (int iMom = 0; iMom < nMom; ++iMom) {
      const double nesOffset = nes.OffsetValue(0, iMom);
      const std::size_t alignedSize =
          static_cast<std::size_t>(nE) * nE * nT * nEta;
      std::vector<double> hostBuf(alignedSize);

      WeakLibReader::PreAlignScatteringKernelMoment(
          nes.KernelData(), nesLayout, energyAxis,
          iMom, nT, nEta,
          opacity_table.energyGrid.values.data(), nE,
          nesOffset, hostBuf.data());

      nes_aligned[iMom].nAlignedE = nE;
      nes_aligned[iMom].nDim3 = nT;
      nes_aligned[iMom].nDim4 = nEta;
      int ext[4] = {nE, nE, nT, nEta};
      nes_aligned[iMom].layout = WeakLibReader::MakeLayout(ext, 4);
      nes_aligned[iMom].data.resize(alignedSize);
      amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                       hostBuf.begin(), hostBuf.end(),
                       nes_aligned[iMom].data.begin());
    }
    CCTK_VINFO("NES pre-aligned: %d moments, %d energy points", nMom, nE);
  }

  // Pre-align Pair
  if (opacity_table.scatPair.IsLoaded()) {
    const auto& pair = opacity_table.scatPair;
    const int nE = opacity_table.energyGrid.nPoints;
    const int nMom = pair.nMoments;
    const int nT = pair.dimensions[3];
    const int nEta = pair.dimensions[4];
    const auto energyAxis = opacity_table.energyGrid.MakeAxis();
    const auto pairLayout = WeakLibReader::MakeLayout(pair.dimensions.data(), 5);

    pair_aligned.resize(nMom);
    for (int iMom = 0; iMom < nMom; ++iMom) {
      const double pairOffset = pair.OffsetValue(0, iMom);
      const std::size_t alignedSize =
          static_cast<std::size_t>(nE) * nE * nT * nEta;
      std::vector<double> hostBuf(alignedSize);

      WeakLibReader::PreAlignScatteringKernelMoment(
          pair.KernelData(), pairLayout, energyAxis,
          iMom, nT, nEta,
          opacity_table.energyGrid.values.data(), nE,
          pairOffset, hostBuf.data());

      pair_aligned[iMom].nAlignedE = nE;
      pair_aligned[iMom].nDim3 = nT;
      pair_aligned[iMom].nDim4 = nEta;
      int ext[4] = {nE, nE, nT, nEta};
      pair_aligned[iMom].layout = WeakLibReader::MakeLayout(ext, 4);
      pair_aligned[iMom].data.resize(alignedSize);
      amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                       hostBuf.begin(), hostBuf.end(),
                       pair_aligned[iMom].data.begin());
    }
    CCTK_VINFO("Pair pre-aligned: %d moments, %d energy points", nMom, nE);
  }

  // Pre-align Brem (nMom=1, dims[3]=nRho, dims[4]=nT)
  if (opacity_table.scatBrem.IsLoaded()) {
    const auto& brem = opacity_table.scatBrem;
    const int nE = opacity_table.energyGrid.nPoints;
    const int nMom = brem.nMoments;
    const int nRho = brem.dimensions[3];
    const int nT = brem.dimensions[4];
    const auto energyAxis = opacity_table.energyGrid.MakeAxis();
    const auto bremLayout = WeakLibReader::MakeLayout(brem.dimensions.data(), 5);

    brem_aligned.resize(nMom);
    for (int iMom = 0; iMom < nMom; ++iMom) {
      const double bremOffset = brem.OffsetValue(0, iMom);
      const std::size_t alignedSize =
          static_cast<std::size_t>(nE) * nE * nRho * nT;
      std::vector<double> hostBuf(alignedSize);

      WeakLibReader::PreAlignScatteringKernelMoment(
          brem.KernelData(), bremLayout, energyAxis,
          iMom, nRho, nT,
          opacity_table.energyGrid.values.data(), nE,
          bremOffset, hostBuf.data());

      brem_aligned[iMom].nAlignedE = nE;
      brem_aligned[iMom].nDim3 = nRho;
      brem_aligned[iMom].nDim4 = nT;
      int ext[4] = {nE, nE, nRho, nT};
      brem_aligned[iMom].layout = WeakLibReader::MakeLayout(ext, 4);
      brem_aligned[iMom].data.resize(alignedSize);
      amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                       hostBuf.begin(), hostBuf.end(),
                       brem_aligned[iMom].data.begin());
    }
    CCTK_VINFO("Brem pre-aligned: %d moments, %d energy points", nMom, nE);
  }

  CCTK_VINFO("Opacity tables loaded: EmAb=%s Iso=%s NES=%s Pair=%s Brem=%s",
             opacity_table_device.HasEmAb() ? "yes" : "no",
             opacity_table_device.HasScatIso() ? "yes" : "no",
             opacity_table_device.HasScatNES() ? "yes" : "no",
             opacity_table_device.HasScatPair() ? "yes" : "no",
             opacity_table_device.HasScatBrem() ? "yes" : "no");
}

extern "C" void TestWeakLibReader_InitEmAb(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestWeakLibReader_InitEmAb;

  if (!opacity_table_device.HasEmAb()) {
    CCTK_INFO("No EmAb table loaded, skipping");
    return;
  }

  CCTK_INFO("Testing EmAb 4D interpolation");

  // Build 4D axes: [E, Rho, T, Ye]
  const WeakLibReader::Axis axes[4] = {
    opacity_table_device.energyGrid.MakeAxis(),    // E (log10)
    opacity_table_device.thermoState.axes[0],      // Rho (log10)
    opacity_table_device.thermoState.axes[1],      // T (log10)
    opacity_table_device.thermoState.axes[2],      // Ye (linear)
  };

  // Fixed energy at midpoint of energy grid (set during load from host data)
  const double fixedE = fixedEnergyMidpoint;

  // Species 0 = electron neutrino
  const double offset = opacity_table_device.emAb.offsets[0];
  const double* opacityData = opacity_table_device.emAb.OpacityData(0);

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      const double rho = RescaleToAxis(p.x, axes[1]);
      const double T   = RescaleToAxis(p.y, axes[2]);
      const double Ye  = RescaleToAxis(p.z, axes[3]);
      opacity_emab(p.I) = WeakLibReader::LogInterpolateSingleVariable4DCustomPoint(
          fixedE, rho, T, Ye,
          axes,
          opacityData, offset);
      });
}

extern "C" void TestWeakLibReader_InitEmAbAnue(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestWeakLibReader_InitEmAbAnue;

  if (!opacity_table_device.HasEmAb()) {
    CCTK_INFO("No EmAb table loaded, skipping antineutrino test");
    return;
  }

  CCTK_INFO("Testing EmAb 4D interpolation (species=1, antineutrino)");

  const WeakLibReader::Axis axes[4] = {
    opacity_table_device.energyGrid.MakeAxis(),
    opacity_table_device.thermoState.axes[0],
    opacity_table_device.thermoState.axes[1],
    opacity_table_device.thermoState.axes[2],
  };

  const double fixedE = fixedEnergyMidpoint;

  // Species 1 = electron antineutrino
  const double offset = opacity_table_device.emAb.offsets[1];
  const double* opacityData = opacity_table_device.emAb.OpacityData(1);

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      const double rho = RescaleToAxis(p.x, axes[1]);
      const double T   = RescaleToAxis(p.y, axes[2]);
      const double Ye  = RescaleToAxis(p.z, axes[3]);
      opacity_emab_anue(p.I) = WeakLibReader::LogInterpolateSingleVariable4DCustomPoint(
          fixedE, rho, T, Ye,
          axes,
          opacityData, offset);
      });
}

extern "C" void TestWeakLibReader_InitIso(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestWeakLibReader_InitIso;

  if (!opacity_table_device.HasScatIso() || iso_slice_device.empty()) {
    CCTK_INFO("No Iso table loaded, skipping");
    return;
  }

  CCTK_INFO("Testing Iso 4D interpolation (moment=0, species=0)");

  // 4D axes: [E, Rho, T, Ye] (same as EmAb after moment extraction)
  const WeakLibReader::Axis axes[4] = {
    opacity_table_device.energyGrid.MakeAxis(),
    opacity_table_device.thermoState.axes[0],
    opacity_table_device.thermoState.axes[1],
    opacity_table_device.thermoState.axes[2],
  };

  // Fixed energy at midpoint of energy grid (set during load from host data)
  const double fixedE = fixedEnergyMidpoint;

  // Offset for species=0, moment=0 from the 2D offset table
  const double offset = opacity_table_device.scatIso.OffsetValue(0, 0);
  const double* sliceData = iso_slice_device.data();

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      const double rho = RescaleToAxis(p.x, axes[1]);
      const double T   = RescaleToAxis(p.y, axes[2]);
      const double Ye  = RescaleToAxis(p.z, axes[3]);
      opacity_iso(p.I) = WeakLibReader::LogInterpolateSingleVariable4DCustomPoint(
          fixedE, rho, T, Ye,
          axes,
          sliceData, offset);
      });
}

extern "C" void TestWeakLibReader_InitNES(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestWeakLibReader_InitNES;

  if (nes_aligned.empty()) {
    CCTK_INFO("No NES aligned table, skipping");
    return;
  }

  CCTK_INFO("Testing NES aligned interpolation (moment=0)");

  // Aligned table for moment 0 (H_I zeroth moment)
  const double* nesData = nes_aligned[0].data.data();
  const WeakLibReader::Layout nesLayout = nes_aligned[0].layout;
  const int nAlignedE = nes_aligned[0].nAlignedE;
  const int iE_mid = nAlignedE / 2;

  // Axes for aligned interpolation: [T, Eta]
  const WeakLibReader::Axis axes[2] = {
    opacity_table_device.thermoState.axes[1],  // T
    opacity_table_device.etaGrid.MakeAxis(),   // Eta
  };

  const double nesOffset = opacity_table_device.scatNES.OffsetValue(0, 0);

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      const double T   = RescaleToAxis(p.y, axes[0]);
      const double eta = eos_log_eta(p.I);

      int idxT = 0, idxEta = 0;
      double fracT = 0.0, fracEta = 0.0;
      WeakLibReader::detail::IndexAndDelta(axes[0], T, idxT, fracT);
      WeakLibReader::detail::IndexAndDelta(axes[1], eta, idxEta, fracEta);

      opacity_nes(p.I) = WeakLibReader::LinearInterp2D4DArray2DAlignedPoint(
          iE_mid, iE_mid,
          idxT, idxEta,
          fracT, fracEta,
          nesOffset,
          nesData, nesLayout);
      });
}

extern "C" void TestWeakLibReader_InitNESOffDiag(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestWeakLibReader_InitNESOffDiag;

  if (nes_aligned.empty()) {
    CCTK_INFO("No NES aligned table, skipping off-diagonal test");
    return;
  }

  CCTK_INFO("Testing NES aligned interpolation (off-diagonal energy pair)");

  const double* nesData = nes_aligned[0].data.data();
  const WeakLibReader::Layout nesLayout = nes_aligned[0].layout;
  const int nAlignedE = nes_aligned[0].nAlignedE;
  if (nAlignedE < 3) {
    CCTK_WARN(1, "NES aligned table has fewer than 3 energy points; skipping off-diagonal test");
    return;
  }
  const int iE_lo = nAlignedE / 2 - 1;
  const int iE_hi = nAlignedE / 2 + 1;

  const WeakLibReader::Axis axes[2] = {
    opacity_table_device.thermoState.axes[1],
    opacity_table_device.etaGrid.MakeAxis(),
  };

  const double nesOffset = opacity_table_device.scatNES.OffsetValue(0, 0);

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      const double T   = RescaleToAxis(p.y, axes[0]);
      const double eta = eos_log_eta(p.I);

      int idxT = 0, idxEta = 0;
      double fracT = 0.0, fracEta = 0.0;
      WeakLibReader::detail::IndexAndDelta(axes[0], T, idxT, fracT);
      WeakLibReader::detail::IndexAndDelta(axes[1], eta, idxEta, fracEta);

      opacity_nes_offdiag(p.I) = WeakLibReader::LinearInterp2D4DArray2DAlignedPoint(
          iE_lo, iE_hi,
          idxT, idxEta,
          fracT, fracEta,
          nesOffset,
          nesData, nesLayout);
      });
}

extern "C" void TestWeakLibReader_InitPair(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestWeakLibReader_InitPair;

  if (pair_aligned.empty()) {
    CCTK_INFO("No Pair aligned table, skipping");
    return;
  }

  CCTK_INFO("Testing Pair aligned interpolation (moment=0)");

  const double* pairData = pair_aligned[0].data.data();
  const WeakLibReader::Layout pairLayout = pair_aligned[0].layout;
  const int nAlignedE = pair_aligned[0].nAlignedE;
  const int iE_mid = nAlignedE / 2;

  // Axes for aligned interpolation: [T, Eta]
  const WeakLibReader::Axis axes[2] = {
    opacity_table_device.thermoState.axes[1],  // T
    opacity_table_device.etaGrid.MakeAxis(),   // Eta
  };

  const double pairOffset = opacity_table_device.scatPair.OffsetValue(0, 0);

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      const double T   = RescaleToAxis(p.y, axes[0]);
      const double eta = eos_log_eta(p.I);

      int idxT = 0, idxEta = 0;
      double fracT = 0.0, fracEta = 0.0;
      WeakLibReader::detail::IndexAndDelta(axes[0], T, idxT, fracT);
      WeakLibReader::detail::IndexAndDelta(axes[1], eta, idxEta, fracEta);

      opacity_pair(p.I) = WeakLibReader::LinearInterp2D4DArray2DAlignedPoint(
          iE_mid, iE_mid,
          idxT, idxEta,
          fracT, fracEta,
          pairOffset,
          pairData, pairLayout);
      });
}

extern "C" void TestWeakLibReader_InitPairOffDiag(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestWeakLibReader_InitPairOffDiag;

  if (pair_aligned.empty()) {
    CCTK_INFO("No Pair aligned table, skipping off-diagonal test");
    return;
  }

  CCTK_INFO("Testing Pair aligned interpolation (off-diagonal energy pair)");

  const double* pairData = pair_aligned[0].data.data();
  const WeakLibReader::Layout pairLayout = pair_aligned[0].layout;
  const int nAlignedE = pair_aligned[0].nAlignedE;
  if (nAlignedE < 3) {
    CCTK_WARN(1, "Pair aligned table has fewer than 3 energy points; skipping off-diagonal test");
    return;
  }
  const int iE_lo = nAlignedE / 2 - 1;
  const int iE_hi = nAlignedE / 2 + 1;

  const WeakLibReader::Axis axes[2] = {
    opacity_table_device.thermoState.axes[1],
    opacity_table_device.etaGrid.MakeAxis(),
  };

  const double pairOffset = opacity_table_device.scatPair.OffsetValue(0, 0);

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      const double T   = RescaleToAxis(p.y, axes[0]);
      const double eta = eos_log_eta(p.I);

      int idxT = 0, idxEta = 0;
      double fracT = 0.0, fracEta = 0.0;
      WeakLibReader::detail::IndexAndDelta(axes[0], T, idxT, fracT);
      WeakLibReader::detail::IndexAndDelta(axes[1], eta, idxEta, fracEta);

      opacity_pair_offdiag(p.I) = WeakLibReader::LinearInterp2D4DArray2DAlignedPoint(
          iE_lo, iE_hi,
          idxT, idxEta,
          fracT, fracEta,
          pairOffset,
          pairData, pairLayout);
      });
}

extern "C" void TestWeakLibReader_InitBrem(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestWeakLibReader_InitBrem;

  if (brem_aligned.empty()) {
    CCTK_INFO("No Brem aligned table, skipping");
    return;
  }

  CCTK_INFO("Testing Brem sum-aligned interpolation");

  const double* bremData = brem_aligned[0].data.data();
  const WeakLibReader::Layout bremLayout = brem_aligned[0].layout;
  const int nAlignedE = brem_aligned[0].nAlignedE;
  const int iE_mid = nAlignedE / 2;

  // Axes for aligned interpolation: [Rho, T]
  const WeakLibReader::Axis axes[2] = {
    opacity_table_device.thermoState.axes[0],  // Rho
    opacity_table_device.thermoState.axes[1],  // T
  };

  const double bremOffset = opacity_table_device.scatBrem.OffsetValue(0, 0);

  // Brem alpha coefficients: [pp, nn, pn]
  constexpr double alpha[3] = {1.0, 1.0, 28.0 / 3.0};

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      const double T = RescaleToAxis(p.y, axes[1]);

      // Density-weighted terms from EOS-derived GFs
      const double dxVals[3] = {
        eos_dxp(p.I),
        eos_dxn(p.I),
        eos_dxpn(p.I),
      };

      int idxT = 0;
      double fracT = 0.0;
      WeakLibReader::detail::IndexAndDelta(axes[1], T, idxT, fracT);

      // Sum over 3 density terms with alpha weights
      double sum = 0.0;
      for (int l = 0; l < 3; ++l) {
        int idxD = 0;
        double fracD = 0.0;
        WeakLibReader::detail::IndexAndDelta(axes[0], dxVals[l], idxD, fracD);

        const double val = WeakLibReader::LinearInterp2D4DArray2DAlignedPoint(
            iE_mid, iE_mid,
            idxD, idxT,
            fracD, fracT,
            bremOffset,
            bremData, bremLayout);
        sum += alpha[l] * val;
      }

      opacity_brem(p.I) = sum;
      });
}

extern "C" void TestWeakLibReader_TestEmAbSweep(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestWeakLibReader_TestEmAbSweep;

  if (!opacity_table_device.HasEmAb()) {
    CCTK_INFO("No EmAb table loaded, skipping sweep test");
    return;
  }

  CCTK_INFO("Testing EmAb energy sweep (LogInterpolateSingleVariable1D3DCustomPoint)");

  const WeakLibReader::Axis axes[4] = {
    opacity_table_device.energyGrid.MakeAxis(),
    opacity_table_device.thermoState.axes[0],
    opacity_table_device.thermoState.axes[1],
    opacity_table_device.thermoState.axes[2],
  };

  const double offset = opacity_table_device.emAb.offsets[0];
  const double* opacityData = opacity_table_device.emAb.OpacityData(0);

  const double* energyValues = opacity_table_device.energyGrid.values.data();
  const int nE = opacity_table_device.energyGrid.nPoints;

  // Stack buffer limit for device kernels. Low-res tables have nE=40.
  constexpr int kMaxE = 128;

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      const double rho = RescaleToAxis(p.x, axes[1]);
      const double T   = RescaleToAxis(p.y, axes[2]);
      const double Ye  = RescaleToAxis(p.z, axes[3]);

      double sweepOut[kMaxE];
      const int actualE = (nE <= kMaxE) ? nE : kMaxE;

      WeakLibReader::LogInterpolateSingleVariable1D3DCustomPoint(
          energyValues, static_cast<std::size_t>(actualE),
          rho, T, Ye,
          axes,
          opacityData, offset,
          sweepOut);

      // Store sum over energy bins as a single scalar
      double sum = 0.0;
      for (int i = 0; i < actualE; ++i) {
        sum += sweepOut[i];
      }
      opacity_emab_sweep(p.I) = sum;
      });
}

extern "C" void TestWeakLibReader_CleanupOpacity(CCTK_ARGUMENTS) {
  CCTK_INFO("Cleaning up opacity tables");
  opacity_table_device = WeakLibReader::WeakLibOpacityTableDevice{};
  iso_slice_device.clear();
  nes_aligned.clear();
  pair_aligned.clear();
  brem_aligned.clear();
}

} // namespace TestWeakLibReader
