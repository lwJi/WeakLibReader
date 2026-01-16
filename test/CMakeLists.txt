add_executable(WeakLibReaderTests
  test_log_interpolate_2d.cpp
  test_log_interpolate_3d.cpp
  test_log_interpolate_4d5d.cpp
  test_log_interpolate_sweep.cpp
  test_log_interpolate_deriv.cpp
  test_log_interpolate_basis.cpp
  test_log_interpolate_kernel.cpp
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
