#include "hip/hip_runtime.h"
/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#pragma once

#include <cuco/detail/utility/cuda.cuh>

#include <cstddef>

namespace cuco {
namespace detail {

CUCO_SUPPRESS_KERNEL_WARNINGS

/**
 * @brief Initializes each slot in the bucket storage to contain `value`.
 *
 * @tparam BucketT Bucket type
 *
 * @param buckets Pointer to flat storage for buckets
 * @param n Number of input buckets
 * @param value Value to which all values in `slots` are initialized
 */
template <typename BucketT>
CUCO_KERNEL void initialize(BucketT* buckets,
                            cuco::detail::index_type n,
                            typename BucketT::value_type value)
{
  auto const loop_stride = cuco::detail::grid_stride();
  auto idx               = cuco::detail::global_thread_id();

  while (idx < n) {
    auto& bucket_slots = *(buckets + idx);
#pragma unroll
    for (auto& slot : bucket_slots) {
      slot = value;
    }
    idx += loop_stride;
  }
}

}  // namespace detail
}  // namespace cuco
