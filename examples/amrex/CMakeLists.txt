# AMReX CUDA Demo Example
# This is a placeholder build configuration for the CUDA demo scaffold

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(WeakLibReaderExamples
  VERSION 0.1.0
  LANGUAGES CXX)

# NOTE: This example requires CUDA support in AMReX
# Uncomment when ready to build

# find_package(AMReX REQUIRED)
#
# add_executable(amrex_interp_demo
#   interp_demo.cpp)
#
# target_link_libraries(amrex_interp_demo
#   PRIVATE
#     WeakLibReader
#     AMReX::amrex)
#
# if(AMReX_GPU_BACKEND STREQUAL "CUDA")
#   set_target_properties(amrex_interp_demo PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON)
# endif()

message(STATUS "AMReX examples are currently placeholders (see examples/amrex/README.md)")
