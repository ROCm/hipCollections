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

# Todo(HIP): change SOURCE_DIR to GITHUB_REPOSITORY and use the correct URL and GIT_TAG
# GITHUB_REPOSITORY NVIDIA/nvbench
# GIT_TAG main
CPMAddPackage(
  NAME nvbench
  GIT_REPOSITORY https://$ENV{GITHUB_USER}:$ENV{GITHUB_PASS}@github.com/AMD-AI/hipBench
  GIT_TAG main
  GIT_SHALLOW TRUE
  EXCLUDE_FROM_ALL YES
)

###################################################################################################
### compiler function #############################################################################
###################################################################################################

###################################################################################################
function(ConfigureBench BENCH_NAME)
    add_executable(${BENCH_NAME} ${ARGN})
    set_target_properties(${BENCH_NAME} PROPERTIES
                                        POSITION_INDEPENDENT_CODE ON
                                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks")
    target_include_directories(${BENCH_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}")
    #target_compile_options(${BENCH_NAME} PRIVATE --expt-extended-lambda --expt-relaxed-constexpr -lineinfo)
    target_link_libraries(${BENCH_NAME} PRIVATE
                                        nvbench::main
                                        pthread
                                        hipco)
endfunction(ConfigureBench)

file(GLOB HIP_SOURCES
  ./*/*/*.hip)

set_source_files_properties(${HIP_SOURCES} 
	                   PROPERTIES LANGUAGE HIP)

###################################################################################################
### benchmark sources #############################################################################
###################################################################################################

###################################################################################################
# - static_set benchmarks -------------------------------------------------------------------------
ConfigureBench(STATIC_SET_BENCH
  hash_table/static_set/contains_bench.hip
  hash_table/static_set/find_bench.hip
  hash_table/static_set/insert_bench.hip
  hash_table/static_set/retrieve_all_bench.hip
  hash_table/static_set/size_bench.hip)

###################################################################################################
# - static_set benchmarks -------------------------------------------------------------------------
ConfigureBench(STATIC_SET_CG_BENCH
hash_table/static_set_cg/contains_bench.hip
hash_table/static_set_cg/insert_bench.hip
hash_table/static_set_cg/retrieve_all_bench.hip
hash_table/static_set_cg/size_bench.hip)


# ###################################################################################################
# # - static_map benchmarks -------------------------------------------------------------------------
ConfigureBench(STATIC_MAP_BENCH
  hash_table/static_map/insert_bench.hip
  hash_table/static_map/find_bench.hip
  hash_table/static_map/contains_bench.hip
  hash_table/static_map/erase_bench.hip)

# ###################################################################################################
# # - static_map benchmarks -------------------------------------------------------------------------
ConfigureBench(STATIC_MAP_TILE_BENCH
  hash_table/static_map_tile/insert_bench.hip
  hash_table/static_map_tile/find_bench.hip
  hash_table/static_map_tile/contains_bench.hip
  hash_table/static_map_tile/erase_bench.hip)

# ###################################################################################################
# # - static_map benchmarks -------------------------------------------------------------------------
ConfigureBench(STATIC_MAP_BLOCK_BENCH
  hash_table/static_map_block/insert_bench.hip
  hash_table/static_map_block/find_bench.hip
  hash_table/static_map_block/contains_bench.hip
  hash_table/static_map_block/erase_bench.hip)

# ###################################################################################################
# - static_multimap benchmarks --------------------------------------------------------------------
ConfigureBench(STATIC_MULTIMAP_BENCH
  hash_table/static_multimap/insert_bench.hip
  hash_table/static_multimap/retrieve_bench.hip
  hash_table/static_multimap/query_bench.hip
  hash_table/static_multimap/count_bench.hip)

# ###################################################################################################
# # - static_multimap benchmarks testing different CG sizes for probing -----------------------------
ConfigureBench(STATIC_MULTIMAP_CG_BENCH
  hash_table/static_multimap_cg/insert_bench.hip
  hash_table/static_multimap_cg/retrieve_bench.hip
  hash_table/static_multimap_cg/query_bench.hip
  hash_table/static_multimap_cg/count_bench.hip
  )

# ###################################################################################################
# - hash function benchmarks ----------------------------------------------------------------------
ConfigureBench(HASH_BENCH
  hash_bench.hip)
