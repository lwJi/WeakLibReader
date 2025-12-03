set(TEST_SOURCES
  test_log_interpolate.cpp)

if (WEAKLIBREADER_ENABLE_HDF5 AND TARGET HDF5::HDF5)
  list(APPEND TEST_SOURCES test_hdf5_loader.cpp)
endif()

add_executable(WeakLibReaderTests ${TEST_SOURCES})

target_include_directories(WeakLibReaderTests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(WeakLibReaderTests
  PRIVATE
    WeakLibReader)

add_test(
  NAME WeakLibReader.LogInterpolation
  COMMAND WeakLibReaderTests)
