cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(WeakLibReader
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(WeakLibReader INTERFACE)

set(AMREX_ROOT "" CACHE PATH "Root directory of the AMReX installation")
if (NOT AMREX_ROOT AND DEFINED ENV{AMREX_ROOT})
  set(AMREX_ROOT "$ENV{AMREX_ROOT}")
endif()
if (NOT AMREX_ROOT AND DEFINED ENV{AMREX_HOME})
  set(AMREX_ROOT "$ENV{AMREX_HOME}")
endif()

find_path(AMREX_INCLUDE_DIR
  NAMES AMReX_GpuQualifiers.H
  HINTS
    ${AMREX_ROOT}
    ${AMREX_ROOT}/include
  PATH_SUFFIXES
    include)

if (NOT AMREX_INCLUDE_DIR)
  message(FATAL_ERROR "Failed to locate AMReX headers. Set AMREX_ROOT (or cache variable AMREX_INCLUDE_DIR) to the AMReX installation path containing AMReX_GpuQualifiers.H.")
endif()

find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

find_library(AMREX_LIBRARY
  NAMES amrex
  HINTS
    ${AMREX_ROOT}
    ${AMREX_ROOT}/lib
    ${AMREX_ROOT}/lib64)
if (NOT AMREX_LIBRARY)
  message(FATAL_ERROR "Failed to locate libamrex. Set AMREX_ROOT to the AMReX installation prefix containing lib/libamrex.")
endif()

target_include_directories(WeakLibReader INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/WeakLibReader/src>
  $<BUILD_INTERFACE:${AMREX_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:WeakLibReader/src>)
target_link_libraries(WeakLibReader INTERFACE
  OpenMP::OpenMP_CXX
  HDF5::HDF5
  ${AMREX_LIBRARY})

enable_testing()

add_subdirectory(test)
