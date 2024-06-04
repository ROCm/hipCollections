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

include(CTest)

###################################################################################################
# - Fetch Catch   ---------------------------------------------------------------------------------

CPMAddPackage(
  NAME Catch2
  GITHUB_REPOSITORY catchorg/Catch2
  VERSION 3.3.0
)

# Header for catch_discover_tests
if(Catch2_ADDED)
    include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
endif()

###################################################################################################
function(ConfigureTest TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_link_libraries(${TEST_NAME} PRIVATE Catch2::Catch2WithMain hipco hip::device)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(${TEST_NAME} PROPERTIES
                                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    target_compile_options(${TEST_NAME} PRIVATE -Wall -Wno-unused-parameter)
    # Todo(HIP): Find equivalents of --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler -Wno-subobject-linkage
    # Todo(HIP): Enable -Wextra
    catch_discover_tests(${TEST_NAME} EXTRA_ARGS --allow-running-no-tests)
endfunction(ConfigureTest)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - utility tests ---------------------------------------------------------------------------------
ConfigureTest(UTILITY_TEST
    utility/extent_test.hip
    utility/storage_test.hip
    utility/fast_int_test.hip
    utility/hash_test.hip)

###################################################################################################
# - static_set tests ------------------------------------------------------------------------------
ConfigureTest(STATIC_SET_TEST
  static_set/capacity_test.hip
  static_set/heterogeneous_lookup_test.hip
  static_set/insert_and_find_test.hip
  static_set/large_input_test.hip
  static_set/retrieve_all_test.hip
  static_set/size_test.hip
  static_set/unique_sequence_test.hip)

###################################################################################################
# - static_map tests ------------------------------------------------------------------------------
ConfigureTest(STATIC_MAP_TEST
    static_map/capacity_test.hip
    static_map/custom_type_test.hip
    static_map/duplicate_keys_test.hip
    static_map/erase_test.hip
    static_map/heterogeneous_lookup_test.hip
    static_map/insert_and_find_test.hip
    static_map/insert_or_assign_test.hip
    static_map/key_sentinel_test.hip
    static_map/shared_memory_test.hip
    static_map/stream_test.hip
    static_map/unique_sequence_test.hip
    static_map/rehash_test.hip)

###################################################################################################
# - static_multimap tests -------------------------------------------------------------------------
ConfigureTest(STATIC_MULTIMAP_TEST
    static_multimap/custom_pair_retrieve_test.hip
    static_multimap/custom_type_test.hip
    static_multimap/heterogeneous_lookup_test.hip
    static_multimap/insert_if_test.hip
    static_multimap/multiplicity_test.hip
    static_multimap/non_match_test.hip
    static_multimap/pair_function_test.hip)
