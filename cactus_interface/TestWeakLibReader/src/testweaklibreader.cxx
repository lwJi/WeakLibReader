#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <loop_device.hxx>

#include <WeakLibReader_Hdf5Loader.hpp>

namespace TestWeakLibReader {

extern "C" void TestWeakLibReader_LoadTable(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  CCTK_INFO("Loading EOS table");

  WeakLibReader::Hdf5Table table;
  const auto status = WeakLibReader::LoadWeakLibEosTable(eos_table_file, "Pressure", table);

  assert(status == WeakLibReader::Hdf5LoadStatus::Success);
}

extern "C" void TestWeakLibReader_Init(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_TestWeakLibReader_Init;

  CCTK_INFO("Initializing grid function");

  // Launch a kernel on the GPU device
  grid.loop_all_device<0, 0, 0>(
      grid.nghostzones, [=](const Loop::PointDesc &p) { energy(p.I) = 0.0; });
}

} // namespace TestWeakLibReader
