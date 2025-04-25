# hipCollections

> [!CAUTION] 
> This release is an *early-access* software technology preview. Running production workloads is *not* recommended.
***

Header-only library of GPU-accelerated, concurrent data structures.
This is a port of the original CUDA version at https://github.com/NVIDIA/cuCollections to HIP in order to enable support for AMD GPUs.

# Requirements
- ROCm and HIP 6.3 or higher (must include `rocthrust-dev` and `hipcub`)
- CMake 3.23.1 or higher (for building the tests)
- git (for getting `libhipcxx`)
- AMD MI100, MI200, MI300, RDNA3/gfx1100 GPU (other architectures are not supported)
- Linux distribution (tested presently with Ubuntu 22.04)

> [!NOTE]
> If `rocthrust-dev` and `hipcub` is not part of your ROCm installation, you can
install them easily in Ubuntu via apt (e.g. sudo apt-get install rocthrust-dev). 

# How to build the tests

To get started, please have a look at the build script we use for CI at `ci/gpu/build_hip.sh`.
As hipCo is a header-only library, you will usually configure your build system to include the hipCo headers.
In order to build some standalone tests, please run the following from the root directory (to build for AMD GPUs):

`mkdir build && cd build && cmake .. && cmake --build .`

**Note**: This will per default build the tests for CDNA architectures (gfx9**) and for wave front size 64. If you like to use hipCo on gfx1100 with wavefront size 32, you will currently have to enable the compile time option `USE_WARP_SIZE_32` and compile explicitly for this architecture *only*:

`cmake -DUSE_WARPSIZE_32=1 -DCMAKE_HIP_ARCHITECTURES=gfx1100 .. && make -j`

It is presently not possible to compile the unit tests at the same time for multiple architectures that use different default wavefront sizes (e.g., gfx90a and gfx1100).

# Current Limitations
- No support for Windows.
- No support for CUDA backend of HIP has been added yet.
- Only the static_set, static_map and static_multimap containers are supported.
- Wavefront sizes 32 and 64 cannot both be used at the same time in a single binary. 
