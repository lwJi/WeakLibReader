#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <loop_device.hxx>

#include <memory>

#include <WeakLibReader_Hdf5Loader.hpp>
#include <WeakLibReader_LogInterpolate.hpp>

namespace TestWeakLibReader {

std::unique_ptr<WeakLibReader::WeakLibEosTable> eos_table;
WeakLibReader::WeakLibEosTableDevice eos_table_device;

extern "C" void TestWeakLibReader_LoadTable(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  CCTK_INFO("Loading EOS table");

  eos_table = std::make_unique<WeakLibReader::WeakLibEosTable>();

  const auto status =
      WeakLibReader::LoadWeakLibEosTableFullParallel(eos_table_file, *eos_table);

  if (status != WeakLibReader::Hdf5LoadStatus::Success) {
    CCTK_ERROR("Failed to load EOS table");
  }

  eos_table_device = WeakLibReader::MakeDeviceCopy(*eos_table);
}

extern "C" void TestWeakLibReader_Cleanup(CCTK_ARGUMENTS) {
  CCTK_INFO("Cleaning up EOS table");
  eos_table.reset();
  eos_table_device = WeakLibReader::WeakLibEosTableDevice{};
}

extern "C" void TestWeakLibReader_Init(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_TestWeakLibReader_Init;

  CCTK_INFO("Initializing grid function");

  const auto &axes = eos_table_device.axes;
  const auto &ipress = eos_table->indices.iPressure;
  const double pressureOffset = eos_table->offsets[ipress];
  const double* pressureData = eos_table_device.VariableData(ipress);
  const double* rhoGrid = axes[0].grid;
  const double* tempGrid = axes[1].grid;
  const double* yeGrid = axes[2].grid;
  const int nRho = axes[0].n;
  const int nTemp = axes[1].n;
  const int nYe = axes[2].n;

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones, [=](const Loop::PointDesc &p) {
      energy(p.I) = WeakLibReader::LogInterpolateSingleVariable3DCustomPoint(
          p.x, p.y, p.z,
          rhoGrid, nRho,
          tempGrid, nTemp,
          yeGrid, nYe,
          pressureData, pressureOffset);
      });
}

} // namespace TestWeakLibReader
