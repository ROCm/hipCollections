#include "hip/hip_runtime.h"
/*
 * Copyright (c) 2022, Jonas Hahnfeld, CERN.
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <utils.hpp>

#include <hipco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>

#include <catch2/catch_template_test_macros.hpp>

static constexpr int Iters = 10'000;

template <typename View>
__global__ void parallel_sum(View v)
{
  for (int i = 0; i < Iters; i++) {
#if __CUDA_ARCH__ < 700
    if constexpr (hipco::detail::is_packable<typename View::value_type>())
#endif
    {
      auto [iter, inserted] = v.insert_and_find(thrust::make_pair(i, 1));
      // for debugging...
      // if (iter->second < 0) {
      //   asm("trap;");
      // }
      if (!inserted) { iter->second += 1; }
    }
#if __CUDA_ARCH__ < 700
    else {
      v.insert(thrust::make_pair(i, gridDim.x * blockDim.x));
    }
#endif
  }
}

TEMPLATE_TEST_CASE_SIG("Parallel insert-or-update",
                       "",
                       ((typename Key, typename Value, int dummy), Key, Value, dummy), //FIXME(hip): dummy fixes ambiguous get_wrapper calls in catch2
                       (int32_t, int32_t, 1),
                       (int32_t, int64_t, 1),
                       (int64_t, int32_t, 1),
                       (int64_t, int64_t, 1))
{
  hipco::empty_key<Key> empty_key_sentinel{-1};
  hipco::empty_value<Value> empty_value_sentinel{-1};
  hipco::static_map<Key, Value> m(10 * Iters, empty_key_sentinel, empty_value_sentinel);

  static constexpr int Blocks  = 1024;
  static constexpr int Threads = 128;
  parallel_sum<<<Blocks, Threads>>>(m.get_device_mutable_view());
  HIPCO_HIP_TRY(hipDeviceSynchronize());

  thrust::device_vector<Key> d_keys(Iters);
  thrust::device_vector<Value> d_values(Iters);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  m.find(d_keys.begin(), d_keys.end(), d_values.begin());

  REQUIRE(hipco::test::all_of(
    d_values.begin(), d_values.end(), [] __device__(Value v) { return v == Blocks * Threads; }));
}
