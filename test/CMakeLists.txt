add_executable(WeakLibReaderTests
  test_log_interpolate.cpp)

target_include_directories(WeakLibReaderTests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(WeakLibReaderTests
  PRIVATE
    WeakLibReader)

add_test(
  NAME WeakLibReader.LogInterpolation
  COMMAND WeakLibReaderTests)
