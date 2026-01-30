#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <loop_device.hxx>

#include <memory>

#include <WeakLibReader_Hdf5Loader.hpp>

namespace TestWeakLibReader {

std::unique_ptr<WeakLibReader::WeakLibEosTable> eos_table;

extern "C" void TestWeakLibReader_LoadTable(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  CCTK_INFO("Loading EOS table");

  eos_table = std::make_unique<WeakLibReader::WeakLibEosTable>();

  const auto status =
      WeakLibReader::LoadWeakLibEosTableFull(eos_table_file, *eos_table);

  if (status != WeakLibReader::Hdf5LoadStatus::Success) {
    CCTK_ERROR("Failed to load EOS table");
  }
}

extern "C" void TestWeakLibReader_Cleanup(CCTK_ARGUMENTS) {
  CCTK_INFO("Cleaning up EOS table");
  eos_table.reset();
}

extern "C" void TestWeakLibReader_Init(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_TestWeakLibReader_Init;

  CCTK_INFO("Initializing grid function");

  grid.loop_all_device<0, 0, 0>(
      grid.nghostzones, [=](const Loop::PointDesc &p) { energy(p.I) = 0.0; });
}

} // namespace TestWeakLibReader
