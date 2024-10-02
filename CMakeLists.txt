#=============================================================================
# Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

# Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/CUCO_RAPIDS.cmake)
  if(DEFINED ENV{RAPIDS_CMAKE_SCRIPT_BRANCH})
    set(RAPIDS_CMAKE_SCRIPT_BRANCH "$ENV{RAPIDS_CMAKE_SCRIPT_BRANCH}")
  else()
    set(RAPIDS_CMAKE_SCRIPT_BRANCH branch-24.06)
  endif()

  set(URL "https://raw.githubusercontent.com/ROCm/rapids-cmake/${RAPIDS_CMAKE_SCRIPT_BRANCH}/RAPIDS.cmake")
  file(DOWNLOAD ${URL}
    ${CMAKE_CURRENT_BINARY_DIR}/CUCO_RAPIDS.cmake
    STATUS DOWNLOAD_STATUS
  )
  list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
  list(GET DOWNLOAD_STATUS 1 ERROR_MESSAGE)

  if(${STATUS_CODE} EQUAL 0)
    message(STATUS "Downloaded 'CUCO_RAPIDS.cmake' successfully!")
  else()
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/CUCO_RAPIDS.cmake)
    # for debuging: message(FATAL_ERROR "Failed to download 'CUCO_RAPIDS.cmake'. URL: ${URL}, Reason: ${ERROR_MESSAGE}")
    message(FATAL_ERROR "Failed to download 'CUCO_RAPIDS.cmake'. Reason: ${ERROR_MESSAGE}")
  endif()
endif()

if(DEFINED ENV{RAPIDS_CMAKE_BRANCH})
  set(rapids-cmake-branch $ENV{RAPIDS_CMAKE_BRANCH})
endif()

include(${CMAKE_CURRENT_BINARY_DIR}/CUCO_RAPIDS.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-hip)
include(rapids-export)
include(rapids-find)


list(APPEND SUPPORTED_ARCHS "gfx90a" "gfx908" "gfx940" "gfx941" "gfx942" 
                            "gfx90a:sramecc+:xnack-" "gfx908:sramecc+:xnack-"
                            "gfx940:sramecc+:xnack-" "gfx941:sramecc+:xnack-"
                            "gfx942:sramecc+:xnack-"
			    "gfx90a:xnack+" "gfx940:xnack+" "gfx941:xnack+"
			    "gfx942:xnack+")
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
rapids_hip_init_architectures(CUCO "NATIVE")

project(CUCO VERSION 0.0.1 LANGUAGES CXX HIP)
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
option(INSTALL_CUCO "Enable CMake install rules for cuco" ${default_build_option_state})

# Write the version header
rapids_cmake_write_version_file(include/cuco/version_config.hpp)

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
    BUILD_EXPORT_SET cuco-exports
    INSTALL_EXPORT_SET cuco-exports
)

###################################################################################################
# - find packages we depend on --------------------------------------------------------------------
rapids_cpm_init()

# prescribe dependencies (must come after `rapids_cpm_init()`)
include(cpm/package_override)
rapids_cpm_package_override(${CMAKE_CURRENT_LIST_DIR}/versions.json)

include(cmake/thirdparty/get_rocthrust.cmake)
include(cmake/thirdparty/get_libhipcxx.cmake)

#Todo(HIP): Ideally, we could use rapids-cmake to configure hipcub
find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")
find_package(hipcub REQUIRED CONFIG PATHS "/opt/rocm/hipcub")

###################################################################################################
# - cuco target   ---------------------------------------------------------------------------------
add_library(cuco INTERFACE)
add_library(cuco::cuco ALIAS cuco)
target_include_directories(cuco INTERFACE
    INTERFACE   $<BUILD_INTERFACE:${CUCO_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>)
#target_link_libraries(cuco INTERFACE libhipcxx::libhipcxx hip::device $<BUILD_INTERFACE:cuco::Thrust>)
#target_link_libraries(cuco INTERFACE libhipcxx::libhipcxx hip::device $<BUILD_INTERFACE::cuco::rocthrust)
#FIXME(hip): Fix target cuco::rocthrust and use CMake generator expression
target_link_libraries(cuco INTERFACE libhipcxx::libhipcxx hip::device roc::rocthrust)
target_compile_features(cuco INTERFACE cxx_std_17 cuda_std_17)
set(CUCO_CG_SIZE "8" CACHE STRING "Size of cooperative groups")
target_compile_options(cuco INTERFACE "-DCUCO_CG_SIZE=${CUCO_CG_SIZE}")
set(CUCO_TILE_SIZE "4" CACHE STRING "Size of cooperative groups")
target_compile_options(cuco INTERFACE "-DCUCO_TILE_SIZE=${CUCO_TILE_SIZE}")
set(CUCO_BLOCK_SIZE "128" CACHE STRING "Size of thread blocks")
target_compile_options(cuco INTERFACE "-DCUCO_BLOCK_SIZE=${CUCO_BLOCK_SIZE}")
set(CUCO_STATIC_SET_CG_SIZE "1" CACHE STRING "Size of cooperative groups for static_set")
target_compile_options(cuco INTERFACE "-DCUCO_STATIC_SET_CG_SIZE=${CUCO_STATIC_SET_CG_SIZE}")
# TODO(HIP): This is a hotfix and does not address the underlying weakness in the code, to
# be revisited later.
target_compile_options(cuco INTERFACE "-Wno-c++11-narrowing-const-reference")
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

install(TARGETS cuco EXPORT cuco-exports)

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

# build directory cuco-config generation
rapids_export(
    BUILD cuco
    EXPORT_SET cuco-exports
    GLOBAL_TARGETS cuco
    NAMESPACE cuco::
    DOCUMENTATION doc_string)

if(INSTALL_CUCO)
    install(DIRECTORY include/cuco/ DESTINATION include/cuco)
    install(DIRECTORY include/hip_extensions/ DESTINATION include/cuco/hip_extensions)
    install(FILES ${CUCO_BINARY_DIR}/include/cuco/version_config.hpp DESTINATION include/cuco)
    # install directory cuco-config generation
    rapids_export(
        INSTALL cuco
        EXPORT_SET cuco-exports
        GLOBAL_TARGETS cuco
        NAMESPACE cuco::
        DOCUMENTATION doc_string)
endif()
