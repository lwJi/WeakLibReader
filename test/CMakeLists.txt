find_package(MPI QUIET)
option(WEAKLIBREADER_ENABLE_MPI_TESTS "Enable MPI-based WeakLibReader tests" OFF)

add_executable(WeakLibReaderTests
  test_log_interpolate.cpp
  test_hdf5_loader.cpp)

target_include_directories(WeakLibReaderTests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(WeakLibReaderTests
  PRIVATE
    WeakLibReader)

add_test(
  NAME WeakLibReader.LogInterpolation
  COMMAND WeakLibReaderTests)

if (WEAKLIBREADER_ENABLE_MPI_TESTS AND MPIEXEC)
  set(_weaklibreader_mpi_command ${MPIEXEC})
  if (MPIEXEC_NUMPROC_FLAG)
    list(APPEND _weaklibreader_mpi_command ${MPIEXEC_NUMPROC_FLAG})
  else()
    list(APPEND _weaklibreader_mpi_command -n)
  endif()
  list(APPEND _weaklibreader_mpi_command 2)
  if (MPIEXEC_PREFLAGS)
    list(APPEND _weaklibreader_mpi_command ${MPIEXEC_PREFLAGS})
  endif()
  list(APPEND _weaklibreader_mpi_command $<TARGET_FILE:WeakLibReaderTests>)
  if (MPIEXEC_POSTFLAGS)
    list(APPEND _weaklibreader_mpi_command ${MPIEXEC_POSTFLAGS})
  endif()
  list(APPEND _weaklibreader_mpi_command "[loader-parallel]")

  add_test(
    NAME WeakLibReader.Hdf5LoaderParallel
    COMMAND ${_weaklibreader_mpi_command})
endif ()
