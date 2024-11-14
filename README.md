 Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.

# hipCollections
Header-only library of GPU-accelerated, concurrent data structures.
This is a port of the original CUDA version at https://github.com/NVIDIA/cuCollections to HIP in order to enable support for AMD GPUs.

# Requirements
- ROCm and HIP 6.0.2 or higher
- CMake 3.23.1 or higher (for building the tests)
- AMD MI100/MI200/MI300 GPU (other architectures are not supported)
- Linux distribution (tested presently with Ubuntu 20.04)

# How to build the tests

To get started, please have a look at the build script we use for CI at `ci/gpu/build_hip.sh`.
As hipCo is a header-only library, you will usually configure your build system to include the hipCo headers.
In order to build some standalone tests, please run the following from the root directory (to build for MI100 + MI200):

`mkdir build && cd build && cmake .. `

# Current Limitations
- No support for Windows.
- No support for CUDA backend of HIP has been added yet.
- Only static_set, static_map and static_multimap containers are supported.
