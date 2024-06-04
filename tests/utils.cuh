#include "hip/hip_runtime.h"
/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#pragma once

namespace hipco {
namespace test {
namespace detail {

template <typename Iterator, typename Predicate>
__global__ void count_if(Iterator begin, Iterator end, int* count, Predicate p)
{
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto it  = begin + tid;

  while (it < end) {
    atomicAdd(count, static_cast<int>(p(*it)));
    it += gridDim.x * blockDim.x;
  }
}

template <typename Iterator1, typename Iterator2, typename Predicate>
__global__ void count_if(
  Iterator1 begin1, Iterator1 end1, Iterator2 begin2, int* count, Predicate p)
{
  auto const n = end1 - begin1;
  auto tid     = blockDim.x * blockIdx.x + threadIdx.x;

  while (tid < n) {
    auto cmp = begin1 + tid;
    auto ref = begin2 + tid;
    atomicAdd(count, static_cast<int>(p(*cmp, *ref)));
    tid += gridDim.x * blockDim.x;
  }
}

}  // namespace detail
}  // namespace test
}  // namespace hipco
