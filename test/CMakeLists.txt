# All test files compiled into a single executable
set(TEST_SOURCES
  test_log_interpolate_2d.cpp
  test_log_interpolate_3d.cpp
  test_log_interpolate_4d5d.cpp
  test_log_interpolate_sweep.cpp
  test_log_interpolate_deriv.cpp
  test_log_interpolate_basis.cpp
  test_log_interpolate_kernel.cpp
  test_hdf5_loader.cpp
  test_weaklib_eos_loader.cpp
  test_weaklib_opacity_loader.cpp
  test_eos_inversion.cpp)

add_executable(WeakLibReaderTests ${TEST_SOURCES})

target_include_directories(WeakLibReaderTests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(WeakLibReaderTests PRIVATE WeakLibReader)

add_test(NAME WeakLibReader.LogInterpolation COMMAND WeakLibReaderTests)
