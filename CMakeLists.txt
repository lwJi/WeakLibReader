cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(WeakLibReader
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(WeakLibReader INTERFACE)

# Allow building AMReX automatically when a pre-installed tree is not provided.
set(AMREX_ROOT "" CACHE PATH "Root directory of the AMReX installation")
set(WeakLibReader_AMREX_GIT_TAG "development" CACHE STRING "Git tag/branch of AMReX to fetch when auto-building")

if (NOT AMREX_ROOT AND DEFINED ENV{AMREX_ROOT})
  set(AMREX_ROOT "$ENV{AMREX_ROOT}")
endif()
if (NOT AMREX_ROOT AND DEFINED ENV{AMREX_HOME})
  set(AMREX_ROOT "$ENV{AMREX_HOME}")
endif()

if (NOT AMREX_ROOT)
  set(_weaklibreader_fetch_amrex_default ON)
else()
  set(_weaklibreader_fetch_amrex_default OFF)
endif()

if (NOT DEFINED WeakLibReader_FETCH_AMREX)
  set(WeakLibReader_FETCH_AMREX ${_weaklibreader_fetch_amrex_default})
endif()
set(WeakLibReader_FETCH_AMREX "${WeakLibReader_FETCH_AMREX}" CACHE BOOL
    "Download and build AMReX automatically if AMREX_ROOT is empty" FORCE)

if (WeakLibReader_FETCH_AMREX)
  include(FetchContent)
  set(_weaklibreader_amrex_install "${CMAKE_BINARY_DIR}/amrex-install")

  FetchContent_Declare(WeakLibReader_AMReX
    GIT_REPOSITORY https://github.com/AMReX-Codes/amrex.git
    GIT_TAG ${WeakLibReader_AMREX_GIT_TAG}
    GIT_SHALLOW TRUE)
  FetchContent_GetProperties(WeakLibReader_AMReX)
  if (NOT WeakLibReader_AMReX_POPULATED)
    message(STATUS "Fetching AMReX from ${WeakLibReader_AMREX_GIT_TAG}")
    FetchContent_Populate(WeakLibReader_AMReX)
  endif()

  set(_weaklibreader_amrex_build "${CMAKE_BINARY_DIR}/amrex-build")
  set(_weaklibreader_amrex_stamp "${_weaklibreader_amrex_build}/WeakLibReader_amrex_stamp")

  if (NOT EXISTS "${_weaklibreader_amrex_stamp}")
    file(MAKE_DIRECTORY "${_weaklibreader_amrex_build}")
    set(_weaklibreader_configure_cmd
        ${CMAKE_COMMAND} -S "${WeakLibReader_AMReX_SOURCE_DIR}" -B "${_weaklibreader_amrex_build}"
        -DCMAKE_INSTALL_PREFIX="${_weaklibreader_amrex_install}"
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DAMReX_MPI=OFF
        -DAMReX_GPU_BACKEND=NONE
        -DAMReX_LINEAR_SOLVERS=OFF
        -DAMReX_PARTICLES=OFF
        -DAMReX_FORTRAN_INTERFACES=OFF
        -DAMReX_SENSEI=OFF
        -DAMReX_BASE_PROFILE=OFF)
    execute_process(COMMAND ${_weaklibreader_configure_cmd}
                    RESULT_VARIABLE _weaklibreader_amrex_configure_result)
    if (_weaklibreader_amrex_configure_result)
      message(FATAL_ERROR "Failed to configure AMReX (error ${_weaklibreader_amrex_configure_result}).")
    endif()

    execute_process(COMMAND ${CMAKE_COMMAND} --build "${_weaklibreader_amrex_build}" --target install --parallel
                    RESULT_VARIABLE _weaklibreader_amrex_build_result)
    if (_weaklibreader_amrex_build_result)
      message(FATAL_ERROR "Failed to build/install AMReX (error ${_weaklibreader_amrex_build_result}).")
    endif()

    file(WRITE "${_weaklibreader_amrex_stamp}" "built")
  endif()

  set(AMREX_ROOT "${_weaklibreader_amrex_install}" CACHE PATH "Root directory of the AMReX installation" FORCE)
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
