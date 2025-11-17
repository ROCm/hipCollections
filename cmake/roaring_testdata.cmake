# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# MIT License
#
# Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Only act if enabled
if(NOT CUCO_DOWNLOAD_ROARING_TESTDATA)
  return()
endif()

set(CUCO_ROARING_DATA_DIR "${CMAKE_BINARY_DIR}/data/roaring_bitmap")

file(MAKE_DIRECTORY "${CUCO_ROARING_DATA_DIR}")

set(ROARING_FORMATSPEC_BASE "https://raw.githubusercontent.com/RoaringBitmap/RoaringFormatSpec/5177ad9")

# TODO(HIP/AMD): Re-add when available in rocmds-cmake
#rapids_cmake_download_with_retry("${ROARING_FORMATSPEC_BASE}/testdata/bitmapwithoutruns.bin"
#                                 "${CUCO_ROARING_DATA_DIR}/bitmapwithoutruns.bin"
#                                 "d719ae2e0150a362ef7cf51c361527585891f01460b1a92bcfb6a7257282a442")

#rapids_cmake_download_with_retry("${ROARING_FORMATSPEC_BASE}/testdata/bitmapwithruns.bin"
#                                 "${CUCO_ROARING_DATA_DIR}/bitmapwithruns.bin"
#                                 "1f1909bfdd354fa2f0694fe88b8076833ca5383ad9fc3f68f2709c84a2ab70e3")

#rapids_cmake_download_with_retry("${ROARING_FORMATSPEC_BASE}/testdata64/portable_bitmap64.bin"
#                                 "${CUCO_ROARING_DATA_DIR}/portable_bitmap64.bin"
#                                 "b5a553a759167f5f9ccb3fa21552d943b4c73235635b753376f4faf62067d178")

message(STATUS "Roaring Bitmap test data downloaded to: ${CUCO_ROARING_DATA_DIR}")

# Define macro only when data is available
add_compile_definitions(CUCO_ROARING_DATA_DIR="${CUCO_ROARING_DATA_DIR}")
