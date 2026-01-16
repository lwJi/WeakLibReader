cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(SET CMP0144 NEW)

project(WeakLibReader
  VERSION 0.1.0
  LANGUAGES C CXX)

# C++ standard settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# AMReX location: check cache, then environment variables
set(AMREX_ROOT "" CACHE PATH "Root directory of the AMReX installation")
if(NOT AMREX_ROOT)
  if(DEFINED ENV{AMREX_ROOT})
    set(AMREX_ROOT "$ENV{AMREX_ROOT}")
  elseif(DEFINED ENV{AMREX_HOME})
    set(AMREX_ROOT "$ENV{AMREX_HOME}")
  endif()
endif()

# Required dependencies
find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

# AMReX: prefer CMake package, fall back to manual detection
find_package(AMReX QUIET CONFIG
  HINTS ${AMREX_ROOT} ${AMREX_ROOT}/lib/cmake ${AMREX_ROOT}/lib64/cmake)

if(NOT AMReX_FOUND)
  # Manual fallback: locate headers and library directly
  find_path(AMREX_INCLUDE_DIR
    NAMES AMReX_GpuQualifiers.H
    HINTS ${AMREX_ROOT}
    PATH_SUFFIXES include)

  if(NOT AMREX_INCLUDE_DIR)
    message(FATAL_ERROR
      "Failed to locate AMReX headers. "
      "Set AMREX_ROOT to the AMReX installation path containing AMReX_GpuQualifiers.H.")
  endif()

  find_library(AMREX_LIBRARY
    NAMES amrex
    HINTS ${AMREX_ROOT}
    PATH_SUFFIXES lib lib64)

  if(NOT AMREX_LIBRARY)
    message(FATAL_ERROR
      "Failed to locate libamrex. "
      "Set AMREX_ROOT to the AMReX installation prefix containing lib/libamrex.")
  endif()

  # Create imported target to match the CMake package interface
  add_library(AMReX::amrex UNKNOWN IMPORTED)
  set_target_properties(AMReX::amrex PROPERTIES
    IMPORTED_LOCATION "${AMREX_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${AMREX_INCLUDE_DIR}")
endif()

# Header-only library target
add_library(WeakLibReader INTERFACE)
target_include_directories(WeakLibReader INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/WeakLibReader/src>
  $<INSTALL_INTERFACE:WeakLibReader/src>)
target_link_libraries(WeakLibReader INTERFACE
  OpenMP::OpenMP_CXX
  HDF5::HDF5
  AMReX::amrex)

# Tests
enable_testing()
add_subdirectory(test)
