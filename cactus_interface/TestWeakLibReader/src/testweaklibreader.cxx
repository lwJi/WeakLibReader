#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <loop_device.hxx>

#include <WeakLibReader_Hdf5Loader.hpp>
#include <WeakLibReader_LogInterpolate.hpp>

namespace TestWeakLibReader {

WeakLibReader::WeakLibEosTableDevice eos_table_device;

extern "C" void TestWeakLibReader_LoadTable(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  CCTK_INFO("Loading EOS table");

  WeakLibReader::WeakLibEosTable eos_table;

  const auto status =
      WeakLibReader::LoadWeakLibEosTableFullParallel(eos_table_file, eos_table);

  if (status != WeakLibReader::Hdf5LoadStatus::Success) {
    CCTK_ERROR("Failed to load EOS table");
  }

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

} // namespace TestWeakLibReader
