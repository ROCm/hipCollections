# hipCollections
Header-only library of GPU-accelerated, concurrent data structures.
This is a port of the original CUDA version at https://github.com/NVIDIA/cuCollections to HIP in order to enable support for AMD GPUs.

# Requirements
- ROCm and HIP 6.2 or higher
- CMake 3.23.1 or higher (for building the tests)
- AMD MI100/MI200/MI300 GPU (other architectures are not supported)
- Linux distribution (tested presently with Ubuntu 20.04)

# How to build the tests

To get started, please have a look at the build script we use for CI at `ci/gpu/build_hip.sh`.
As hipCo is a header-only library, you will usually configure your build system to include the hipCo headers.
In order to build some standalone tests, please run the following from the root directory (to build for AMD GPUs):

`mkdir build && cd build && cmake .. `

# Current Limitations
- No support for Windows.
- No support for CUDA backend of HIP has been added yet.
- Only static_set, static_map and static_multimap containers are supported.
