// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef AMD_COOPERATIVE_GROUPS_EXT
#define AMD_COOPERATIVE_GROUPS_EXT

#include "amd_warp_primitives.h"
#include <hip/device_functions.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#ifdef __AMDGCN_WAVEFRONT_SIZE
#undef WAVEFRONT_SIZE
#define WAVEFRONT_SIZE __AMDGCN_WAVEFRONT_SIZE
#endif

#if !(__gfx1010__ || __gfx1011__ || __gfx1012__ || __gfx1030__ || __gfx1031__)
#if WAVEFRONT_SIZE != 64
#error "WAVEFRONT_SIZE 64 required"
#endif
#endif
using namespace hip_warp_primitives;
namespace hip_cooperative_groups_ext {

class cooperative_groups_based_warp_primitives {
 private:
  uint32_t __size;
  lane_mask __group_mask;

 public:
  __device__ cooperative_groups_based_warp_primitives(uint32_t s, lane_mask m)
  {
    __size       = s;
    __group_mask = m;
  }

  __device__ void set_size(uint32_t s) { __size = s; }

  __device__ uint32_t size() const { return __size; }

  __device__ lane_mask get_mask() const { return __group_mask; }

  __device__ void set_mask(lane_mask lm) { __group_mask = lm; }

  __device__ inline bool any(int pred) const
  {
    assert(__is_thread_in_mask(__group_mask));
    return __any_sync(__group_mask, pred);
  }

  __device__ inline lane_mask ballot(int pred) const
  {
    auto result_ballot_sync = __ballot_sync(__group_mask, pred);
    result_ballot_sync      = (__size == WAVEFRONT_SIZE)
                                ? result_ballot_sync
                                : result_ballot_sync >> __fns64(__group_mask, 0, 1);
    return result_ballot_sync;
  }

  __device__ inline int thread_rank() const
  {
    auto lane_id = __lane_id();
    int rank =
      (__size == WAVEFRONT_SIZE) ? lane_id : __popcll(__group_mask & ((1L << (lane_id)) - 1));
    // printf("mask: %llx lane_id: %d rank %d size %d\n", __group_mask, lane_id, rank, __size);
    return rank;
  }

  __device__ inline void sync() const { return __sync_active_threads(); }

  template <class T>
  __device__ inline T shfl(T var, int srcRank) const
  {
    int srcLane = (__size == WAVEFRONT_SIZE) ? srcRank : __fns64(__group_mask, 0, srcRank + 1);
    // printf("mask %llx rank: %ld lane: %ld\n", __group_mask, srcRank, srcLane);
    return __shfl_sync(__group_mask, var, srcLane);
  }

  __device__ inline void compute_groups()
  {
    lane_mask __group_mask =
      __match_any_sync(get_mask(), threadIdx.x / size());  // pass __group_mask instead of ~0
    set_mask(__group_mask);
  }
};

template <uint32_t CGSIZE>
class tiled_partition_internal_ext : public cooperative_groups_based_warp_primitives {
 public:
  __device__ tiled_partition_internal_ext()  // cooperative_groups::tiled_group& parent
    : cooperative_groups_based_warp_primitives(CGSIZE, ~0)
  {                                          // Include all threads
    compute_groups();
  }
  __device__ inline void compute_groups()
  {
    lane_mask __group_mask =
      __match_any_sync(get_mask(), threadIdx.x / size());  // pass __group_mask instead of ~0
    set_mask(__group_mask);
  }
};

class coalesced_group_ext : public cooperative_groups_based_warp_primitives {
 public:
  __device__ coalesced_group_ext(lane_mask lm)
    : cooperative_groups_based_warp_primitives(__popcll(lm), lm)
  {
    set_mask(lm);
  }
};

template <uint32_t CGSIZE>
__device__ inline coalesced_group_ext binary_partition(
  tiled_partition_internal_ext<CGSIZE>& parent_g, bool pred)
{
  lane_mask pred_mask = __ballot(pred);
  if (pred) {
    return coalesced_group_ext(pred_mask & parent_g.get_mask());
  } else {
    return coalesced_group_ext(~(pred_mask & parent_g.get_mask()));
  }
}

template <uint32_t CGSIZE>
class thread_block_tile : public tiled_partition_internal_ext<CGSIZE> {
 public:
  __device__ thread_block_tile() : tiled_partition_internal_ext<CGSIZE>() {}
};

template <uint32_t CGSIZE>
__device__ thread_block_tile<CGSIZE> tiled_partition(cooperative_groups::thread_block tb)
{
  return thread_block_tile<CGSIZE>();
}

__device__ cooperative_groups::thread_block this_thread_block()
{
  // Todo(HIP): complete the implementation
  return cooperative_groups::this_thread_block();
}

}  // namespace hip_cooperative_groups_ext

#endif