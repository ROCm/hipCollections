#=============================================================================
# Copyright (c) 2018-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/HIPCO_RAPIDS.cmake)
  if (DEFINED ENV{RAPIDS_CMAKE_BRANCH})
    set(RAPIDS_CMAKE_BRANCH "$ENV{RAPIDS_CMAKE_BRANCH}")
  else()
    set(RAPIDS_CMAKE_BRANCH hipdf-dev)
  endif()

  set(URL "https://github.com/ROCm/rapids-cmake/${RAPIDS_CMAKE_BRANCH}/RAPIDS.cmake")
  file(DOWNLOAD ${URL}
       ${CMAKE_CURRENT_BINARY_DIR}/HIPCO_RAPIDS.cmake
       STATUS DOWNLOAD_STATUS
  )
  list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
  list(GET DOWNLOAD_STATUS 1 ERROR_MESSAGE)

  if(${STATUS_CODE} EQUAL 0)
    message(STATUS "Downloaded 'HIPCO_RAPIDS.cmake' successfully!")
  else()
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/HIPCO_RAPIDS.cmake)
    # for debuging: message(FATAL_ERROR "Failed to download 'HIPCO_RAPIDS.cmake'. URL: ${URL}, Reason: ${ERROR_MESSAGE}")
    message(FATAL_ERROR "Failed to download 'HIPCO_RAPIDS.cmake'. Reason: ${ERROR_MESSAGE}")
  endif()
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/HIPCO_RAPIDS.cmake)


include(rapids-cmake)
include(rapids-cpm)
include(rapids-hip)
include(rapids-export)
include(rapids-find)


list(APPEND SUPPORTED_ARCHS "gfx90a" "gfx908" "gfx940" "gfx941" "gfx942" 
                            "gfx90a:sramecc+:xnack-" "gfx908:sramecc+:xnack-"
                            "gfx940:sramecc+:xnack-" "gfx941:sramecc+:xnack-"
                            "gfx942:sramecc+:xnack-")
# WAR for not supporting all architectures yet
# build only for supported architectures per default
if(NOT (DEFINED CMAKE_HIP_ARCHITECTURES))
  set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx908;gfx940;gfx941;gfx942")
else()
  foreach(arch ${CMAKE_HIP_ARCHITECTURES})
    if(NOT ${arch} IN_LIST SUPPORTED_ARCHS) 
      message(FATAL_ERROR "Aborting, as the provided architecture ${arch} is not supported")
    endif()
  endforeach()
endif()

# * Enable the HIP language
rapids_hip_init_architectures(HIPCO)

project(HIPCO VERSION 0.0.1 LANGUAGES CXX HIP)
###################################################################################################
# - build options ---------------------------------------------------------------------------------
set(default_build_option_state OFF)
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  set(default_build_option_state ON)
endif()

option(BUILD_TESTS "Configure CMake to build tests" ON)
#Todo(hip): Enable building benchmarks, NVbench is required
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" ON)
option(BUILD_EXAMPLES "Configure CMake to build examples" ${default_build_option_state})
option(INSTALL_HIPCO "Enable CMake install rules for hipco" ${default_build_option_state})

# Write the version header
rapids_cmake_write_version_file(include/hipco/version_config.hpp)

##############################################################################
# - build type ---------------------------------------------------------------
# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# needed for clangd and clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################################
# - compiler options ---------------------------------------------------------

# * Find HIP package
# * Offers support for CMAKE_CUDA_ARCHITECTURES=NATIVE
rapids_find_package(
    HIP REQUIRED
    BUILD_EXPORT_SET hipco-exports
    INSTALL_EXPORT_SET hipco-exports
)

###################################################################################################
# - find packages we depend on --------------------------------------------------------------------
rapids_cpm_init()

include(cmake/thirdparty/get_rocthrust.cmake)
include(cmake/thirdparty/get_libhipcxx.cmake)

#Todo(HIP): Ideally, we could use rapids-cmake to configure hipcub
find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")
find_package(hipcub REQUIRED CONFIG PATHS "/opt/rocm/hipcub")

###################################################################################################
# - hipco target   ---------------------------------------------------------------------------------
add_library(hipco INTERFACE)
add_library(hipco::hipco ALIAS hipco)
target_include_directories(hipco INTERFACE
    INTERFACE   $<BUILD_INTERFACE:${HIPCO_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>)
#target_link_libraries(hipco INTERFACE libhipcxx::libhipcxx hip::device $<BUILD_INTERFACE:hipco::Thrust>)
#target_link_libraries(hipco INTERFACE libhipcxx::libhipcxx hip::device $<BUILD_INTERFACE::hipco::rocthrust)
#FIXME(hip): Fix target hipco::rocthrust and use CMake generator expression
target_link_libraries(hipco INTERFACE libhipcxx::libhipcxx hip::device roc::rocthrust)
target_compile_features(hipco INTERFACE cxx_std_17 cuda_std_17)
set(HIPCO_CG_SIZE "8" CACHE STRING "Size of cooperative groups")
target_compile_options(hipco INTERFACE "-DHIPCO_CG_SIZE=${HIPCO_CG_SIZE}")
set(HIPCO_TILE_SIZE "4" CACHE STRING "Size of cooperative groups")
target_compile_options(hipco INTERFACE "-DHIPCO_TILE_SIZE=${HIPCO_TILE_SIZE}")
set(HIPCO_BLOCK_SIZE "128" CACHE STRING "Size of thread blocks")
target_compile_options(hipco INTERFACE "-DHIPCO_BLOCK_SIZE=${HIPCO_BLOCK_SIZE}")
set(HIPCO_STATIC_SET_CG_SIZE "1" CACHE STRING "Size of cooperative groups for static_set")
target_compile_options(hipco INTERFACE "-DHIPCO_STATIC_SET_CG_SIZE=${HIPCO_STATIC_SET_CG_SIZE}")
###################################################################################################
# - optionally build tests ------------------------------------------------------------------------
if(BUILD_TESTS)
    add_subdirectory(tests)
endif(BUILD_TESTS)

###################################################################################################
# - Optionally build google benchmarks ------------------------------------------------------------
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif(BUILD_BENCHMARKS)

###################################################################################################
# - Optionally build examples ---------------------------------------------------------------------
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif(BUILD_EXAMPLES)

###################################################################################################
# - Install targets -------------------------------------------------------------------------------
install(TARGETS hipco EXPORT hipco-exports)

set(doc_string
    [=[
Provide targets for hipCollections.

hipCollections (hipco) is an open-source, header-only library of GPU-accelerated,
concurrent data structures. 

Similar to how Thrust and hipCUB provide STL-like, GPU accelerated algorithms and
primitives, hipCollections provides STL-like concurrent data structures.
hipCollections is not a one-to-one, drop-in replacement for STL data structures
like std::unordered_map. Instead, it provides functionally similar data
structures tailored for efficient use with GPUs.

]=])

set(code_string
[=[
if(NOT TARGET hipco::rocthrust)
thrust_create_target(hipco::rocthrust FROM_OPTIONS)
endif()
]=])

# build directory hipco-config generation
rapids_export(
    BUILD hipco
    EXPORT_SET hipco-exports
    GLOBAL_TARGETS hipco
    NAMESPACE hipco::
    DOCUMENTATION doc_string
    FINAL_CODE_BLOCK code_string)

if(INSTALL_HIPCO)
    install(DIRECTORY include/hipco/ DESTINATION include/hipco)
    install(DIRECTORY include/hip_extensions/ DESTINATION include/hip_extensions)
    install(FILES ${HIPCO_BINARY_DIR}/include/hipco/version_config.hpp DESTINATION include/hipco)
    # install directory hipco-config generation
    rapids_export(
        INSTALL hipco
        EXPORT_SET hipco-exports
        GLOBAL_TARGETS hipco
        NAMESPACE hipco::
        DOCUMENTATION doc_string)
endif()
