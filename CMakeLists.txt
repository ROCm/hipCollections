#=============================================================================
# Copyright (c) 2018-2025, NVIDIA CORPORATION.
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

# Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
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

# NOTE(HIP/AMD): make *.cu files HIP lang files
set(CMAKE_USER_MAKE_RULES_OVERRIDE "${CMAKE_CURRENT_LIST_DIR}/overrides.cmake")

# Set default HIP architectures to maintain compatibility with existing scripts
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx942")
endif()

set(
  RAPIDS_CMAKE_MODULE_PATH
  "$ENV{RAPIDS_CMAKE_MODULE_PATH}"
  CACHE FILEPATH
  "Announce that ROCmDS-CMake is available via the provided module path."
)
if (NOT "${RAPIDS_CMAKE_MODULE_PATH}" STREQUAL "")
  list(APPEND CMAKE_MODULE_PATH "${RAPIDS_CMAKE_MODULE_PATH}")
  include(rapids-cmake)  # ensure rapids-cmake-dir is set
  # NOTE(HIP/AMD): Only the installed ROCmDS-CMake package defines
  # rapids-cmake-dir, a source checkout does not.
  if(NOT DEFINED rapids-cmake-dir)
    set(rapids-cmake-dir "${RAPIDS_CMAKE_MODULE_PATH}")
  endif()
else()
  if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/CUCO_RAPIDS.cmake)
    if(DEFINED ENV{RAPIDS_CMAKE_SCRIPT_REPO})
      set(RAPIDS_CMAKE_SCRIPT_REPO "$ENV{RAPIDS_CMAKE_SCRIPT_REPO}")
    else()
      set(RAPIDS_CMAKE_SCRIPT_REPO ROCm-DS/ROCmDS-CMake)
    endif()
    if(DEFINED ENV{RAPIDS_CMAKE_SCRIPT_BRANCH})
      set(RAPIDS_CMAKE_SCRIPT_BRANCH "$ENV{RAPIDS_CMAKE_SCRIPT_BRANCH}")
    else()
      set(RAPIDS_CMAKE_SCRIPT_BRANCH release/rocmds-26.03)
    endif()

    set(URL "https://raw.githubusercontent.com/${RAPIDS_CMAKE_SCRIPT_REPO}/${RAPIDS_CMAKE_SCRIPT_BRANCH}/RAPIDS.cmake")
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
endif()

include(rapids-cmake)
include(rapids-cpm)
include(rapids-hip)
include(rapids-export)
include(rapids-find)

# * Enable the HIP language
rapids_hip_init_architectures(CUCO "NATIVE")

project(CUCO VERSION 0.5.0 LANGUAGES CXX HIP)
###################################################################################################
# - build options ---------------------------------------------------------------------------------
set(default_build_option_state OFF)
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  set(default_build_option_state ON)
endif()

option(BUILD_TESTS "Configure CMake to build tests" ${default_build_option_state})
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" ${default_build_option_state})
option(BUILD_EXAMPLES "Configure CMake to build examples" ${default_build_option_state})
option(INSTALL_CUCO "Enable CMake install rules for cuco" ${default_build_option_state})
option(INSTALL_CUCO_CPM_DEPS "Whether to install/package third-party dependencies of cuco together with cuco. This has only an effect on packages that are downloaded by CPM, not those that CPM finds locally." ON)

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

include(cmake/thirdparty/get_cccl.cmake)

###################################################################################################
# - cuco target   ---------------------------------------------------------------------------------
add_library(cuco INTERFACE)
add_library(cuco::cuco ALIAS cuco)
target_include_directories(cuco INTERFACE
    INTERFACE   $<BUILD_INTERFACE:${CUCO_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>)
target_link_libraries(cuco INTERFACE CCCL::CCCL hip::device)
target_compile_features(cuco INTERFACE cxx_std_17 cuda_std_17)
set(CUCO_CG_SIZE "8" CACHE STRING "Size of cooperative groups")
target_compile_options(cuco INTERFACE "-DCUCO_CG_SIZE=${CUCO_CG_SIZE}")
set(CUCO_TILE_SIZE "4" CACHE STRING "Size of cooperative groups")
target_compile_options(cuco INTERFACE "-DCUCO_TILE_SIZE=${CUCO_TILE_SIZE}")
set(CUCO_BLOCK_SIZE "128" CACHE STRING "Size of thread blocks")
target_compile_options(cuco INTERFACE "-DCUCO_BLOCK_SIZE=${CUCO_BLOCK_SIZE}")
set(CUCO_STATIC_SET_CG_SIZE "4" CACHE STRING "Size of cooperative groups for static_set")
target_compile_options(cuco INTERFACE "-DCUCO_STATIC_SET_CG_SIZE=${CUCO_STATIC_SET_CG_SIZE}")
target_compile_options(cuco INTERFACE "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1")
target_compile_options(cuco INTERFACE "-DHIP_ENABLE_EXTRA_WARP_SYNC_TYPES=1")

if(USE_WARPSIZE_32)
  target_compile_options(cuco INTERFACE "-DCUCO_USE_WARPSIZE_32")
endif()
# TODO(HIP): This is a hotfix and does not address the underlying weakness in the code, to
# be revisited later.
target_compile_options(cuco INTERFACE "-Wno-c++11-narrowing-const-reference")
###################################################################################################
# - Optionally download RoaringFormatSpec test data -----------------------------------------------

option(CUCO_DOWNLOAD_ROARING_TESTDATA "Download RoaringFormatSpec test data" ON)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/roaring_testdata.cmake)

###################################################################################################
# - common compile options function ---------------------------------------------------------------

function(cuco_set_common_compile_options target_name)
    # Parse optional arguments
    cmake_parse_arguments(CUCO_OPTS "ADD_LINEINFO" "" "" ${ARGN})
    
    # NOTE(HIP/AMD): hipcc/clang use plain compiler flags, not nvcc-style
    # `--compiler-options=` / `--expt-extended-lambda` / `-Werror=all-warnings`.
    # -Werror is intentionally omitted: it's brittle across ROCm/clang versions
    # (a new warning in a future toolchain would break the build).
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wno-unused-parameter)

    if(CUCO_OPTS_ADD_LINEINFO)
        target_compile_options(${target_name} PRIVATE -gline-tables-only)
    endif()
endfunction()

###################################################################################################
# - optionally build tests ------------------------------------------------------------------------

if(BUILD_TESTS)
    include(CTest)
    add_subdirectory(tests)
    # Include header tests as part of the regular test suite
    include(cmake/header_testing.cmake)
endif(BUILD_TESTS)

###################################################################################################
# - Optionally build nvbench benchmarks -----------------------------------------------------------

if(BUILD_BENCHMARKS)
    include(${rapids-cmake-dir}/cpm/nvbench.cmake)
    include(${rapids-cmake-dir}/cpm/package_override.cmake)
    rapids_cpm_nvbench(BUILD_STATIC)

    add_subdirectory(benchmarks)
endif(BUILD_BENCHMARKS)

###################################################################################################
# - Optionally build examples ---------------------------------------------------------------------

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif(BUILD_EXAMPLES)


###################################################################################################
# - Install targets -------------------------------------------------------------------------------

install(TARGETS cuco EXPORT cuco-exports COMPONENT devel)

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
  install(DIRECTORY include/cuco/ DESTINATION include/cuco COMPONENT devel)
  install(DIRECTORY include/hip_extensions/ DESTINATION include/hip_extensions COMPONENT devel)
  install(FILES ${CUCO_BINARY_DIR}/include/cuco/version_config.hpp DESTINATION include/cuco COMPONENT devel)
  # install directory cuco-config generation
  rapids_export(
      INSTALL cuco
      EXPORT_SET cuco-exports
      GLOBAL_TARGETS cuco
      NAMESPACE cuco::
      DOCUMENTATION doc_string
      PROJECT_COMPONENT devel)

  set(CPACK_PACKAGE_NAME "hipcollections")
  set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "${CPACK_PACKAGE_VENDOR}")
  set(CPACK_OUTPUT_FILE_PREFIX "dist" CACHE PATH "Package output file prefix.")
  set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
  set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
  set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
  if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
  endif()

  if(NOT INSTALL_CUCO_CPM_DEPS)
    # NOTE: Enable component-based packaging
    #       so that we can filter out the auto-generated
    #       hipccl dependency, which has its own component.
    set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME devel) # note: removes 'Unspecified' component
    set(CPACK_COMPONENTS_ALL devel)
    set(CPACK_DEB_COMPONENT_INSTALL ON)
    set(CPACK_RPM_COMPONENT_INSTALL ON)
    set(CPACK_DEBIAN_DEVEL_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-dev")
    set(CPACK_RPM_DEVEL_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-devel")
    set(CPACK_DEBIAN_DEVEL_PACKAGE_DEPENDS "libhipcxx-dev (>= 2.7.0)")
    set(CPACK_RPM_DEVEL_PACKAGE_REQUIRES "libhipcxx-devel >= 2.7.0")
  else()
    # NOTE: The standalone package ships every component, so packaging stays
    #       monolithic. The per-component name variables above only apply to
    #       component-based packaging and would be ignored here.
    set(CPACK_DEBIAN_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-standalone")
    set(CPACK_RPM_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-standalone")
  endif()
  set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
  set(CPACK_RPM_FILE_NAME RPM-DEFAULT)

  include(CPack)
endif()

