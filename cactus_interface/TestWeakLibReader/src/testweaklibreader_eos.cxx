#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <loop_device.hxx>

#include <cmath>

#include <hdf5/WeakLibReader_Hdf5Loader.hpp>
#include <interp/WeakLibReader_EosInversion.hpp>
#include <interp/WeakLibReader_LogInterpolate.hpp>

namespace TestWeakLibReader {

WeakLibReader::WeakLibEosTableDevice eos_table_device;
WeakLibReader::EosInversionBounds eos_inversion_bounds;

extern "C" void TestWeakLibReader_LoadTable(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  CCTK_INFO("Loading EOS table");

  WeakLibReader::WeakLibEosTable eos_table;

  const auto status =
      WeakLibReader::LoadWeakLibEosTableFullParallel(eos_table_file, eos_table);

  if (status != WeakLibReader::Hdf5LoadStatus::Success) {
    CCTK_ERROR("Failed to load EOS table");
  }

  // Compute inversion bounds while host table is still available
  const int iE = eos_table.indices.iInternalEnergyDensity;
  const int iP = eos_table.indices.iPressure;
  const int iS = eos_table.indices.iEntropyPerBaryon;
  const int totalSize = eos_table.dimensions[0]
                      * eos_table.dimensions[1]
                      * eos_table.dimensions[2];

  eos_inversion_bounds = WeakLibReader::InitializeEosInversionBounds(
      eos_table.axes, totalSize,
      (iE >= 0) ? eos_table.VariableData(iE) : nullptr, (iE >= 0) ? eos_table.offsets[iE] : 0.0,
      (iP >= 0) ? eos_table.VariableData(iP) : nullptr, (iP >= 0) ? eos_table.offsets[iP] : 0.0,
      (iS >= 0) ? eos_table.VariableData(iS) : nullptr, (iS >= 0) ? eos_table.offsets[iS] : 0.0);

  eos_table_device = WeakLibReader::MakeDeviceCopy(eos_table);
}

extern "C" void TestWeakLibReader_Cleanup(CCTK_ARGUMENTS) {
  CCTK_INFO("Cleaning up EOS table");
  eos_table_device = WeakLibReader::WeakLibEosTableDevice{};
}

extern "C" void TestWeakLibReader_Init(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_TestWeakLibReader_Init;

  CCTK_INFO("Initializing grid function");

  const WeakLibReader::Axis axes[3] = {
    eos_table_device.axes[0],
    eos_table_device.axes[1],
    eos_table_device.axes[2],
  };
  const int ipress = eos_table_device.indices.iPressure;
  const double pressureOffset = eos_table_device.offsets[ipress];
  const double* pressureData = eos_table_device.VariableData(ipress);

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      energy(p.I) = WeakLibReader::LogInterpolateSingleVariable3DCustomPoint(
          p.x, p.y, p.z,
          axes,
          pressureData, pressureOffset);
      });
}

extern "C" void TestWeakLibReader_ComputeEosDerived(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestWeakLibReader_ComputeEosDerived;

  if (eos_table_device.nVariables == 0) {
    CCTK_INFO("No EOS table loaded, skipping EOS-derived computation");
    return;
  }

  CCTK_INFO("Computing EOS-derived quantities");

  const WeakLibReader::Axis axes[3] = {
    eos_table_device.axes[0],  // Rho
    eos_table_device.axes[1],  // T
    eos_table_device.axes[2],  // Ye
  };

  // EOS variable indices
  const int iMuE = eos_table_device.indices.iElectronChemicalPotential;
  const int iXp  = eos_table_device.indices.iProtonMassFraction;
  const int iXn  = eos_table_device.indices.iNeutronMassFraction;

  const double* muEData = eos_table_device.VariableData(iMuE);
  const double muEOffset = eos_table_device.offsets[iMuE];
  const double* xpData = eos_table_device.VariableData(iXp);
  const double xpOffset = eos_table_device.offsets[iXp];
  const double* xnData = eos_table_device.VariableData(iXn);
  const double xnOffset = eos_table_device.offsets[iXn];

  // Boltzmann constant in MeV/K (same units as EOS table)
  constexpr double kB_MeV_per_K = 8.6173303e-11;  // MeV/K

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      // (p.x, p.y, p.z) = (rho, T, Ye) in raw units
      const double rho = p.x;
      const double T   = p.y;
      const double Ye  = p.z;

      // Interpolate mu_e from EOS
      const double muE = WeakLibReader::LogInterpolateSingleVariable3DCustomPoint(
          rho, T, Ye, axes, muEData, muEOffset);

      // eta = mu_e / (kB * T) - dimensionless
      // Note: table T is in Kelvin, mu_e in MeV
      const double eta = muE / (kB_MeV_per_K * T);
      eos_log_eta(p.I) = eta;  // raw value; IndexAndDelta handles log

      // Interpolate mass fractions
      const double Xp = WeakLibReader::LogInterpolateSingleVariable3DCustomPoint(
          rho, T, Ye, axes, xpData, xpOffset);
      const double Xn = WeakLibReader::LogInterpolateSingleVariable3DCustomPoint(
          rho, T, Ye, axes, xnData, xnOffset);

      // Density-weighted terms for Brem (raw values)
      eos_dxp(p.I)  = rho * Xp;
      eos_dxn(p.I)  = rho * Xn;
      const double XpXn = Xp * Xn;
      eos_dxpn(p.I) = (XpXn > 0.0) ? rho * sqrt(XpXn) : 0.0;
      });
}

extern "C" void TestWeakLibReader_InvertEos(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestWeakLibReader_InvertEos;

  CCTK_INFO("Testing EOS inversion: recovering T from (rho, E, Ye)");

  const WeakLibReader::Axis axes[3] = {
    eos_table_device.axes[0],  // Rho
    eos_table_device.axes[1],  // T
    eos_table_device.axes[2],  // Ye
  };
  const WeakLibReader::Layout layout = eos_table_device.layout;

  const int iE = eos_table_device.indices.iInternalEnergyDensity;
  const double* eData = eos_table_device.VariableData(iE);
  const double eOffset = eos_table_device.offsets[iE];

  const WeakLibReader::EosInversionBounds bounds = eos_inversion_bounds;

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      const double rho = p.x;
      const double T   = p.y;  // known temperature
      const double Ye  = p.z;

      // Forward interpolate energy
      const double E = WeakLibReader::LogInterpolateSingleVariable3DCustomPoint(
          rho, T, Ye, axes, eData, eOffset);

      // Invert to recover T
      double T_recovered = 0.0;
      const int error = WeakLibReader::ComputeTemperatureFromEnergy(
          rho, E, Ye, axes, eData, layout, eOffset, bounds, T_recovered);

      eos_inv_temp(p.I)  = T_recovered;
      eos_inv_error(p.I) = static_cast<double>(error);
      });
}

} // namespace TestWeakLibReader
