/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <utils.cuh>

#include <hipco/detail/error.hpp>

#include <thrust/functional.h>

#include <hip/hip_cooperative_groups.h>
//Todo(HIP/AMD): Remove the header once we have CG workaround implemented in ROCm
#include <hip_extensions/hip_cooperative_groups_ext/amd_cooperative_groups_ext.cuh>

namespace hipco {
namespace test {
//Todo(HIP/AMD): Change to "namespace cg = cooperative_groups;" once we have CG workaround implemented in ROCm
namespace cg = hip_extensions::hip_cooperative_groups_ext;

constexpr int32_t block_size = 128;

enum class probe_sequence { linear_probing, double_hashing, coalesced_probing };

// User-defined logical algorithms to reduce compilation time
template <typename Iterator, typename Predicate>
int count_if(Iterator begin, Iterator end, Predicate p, hipStream_t stream = 0)
{
  auto const size      = std::distance(begin, end); //end - begin;
  auto const grid_size = (size + block_size - 1) / block_size;

  int* count;
  HIPCO_HIP_TRY(hipMallocManaged(&count, sizeof(int)));

  *count = 0;
  int device_id;
  HIPCO_HIP_TRY(hipGetDevice(&device_id));
  HIPCO_HIP_TRY(hipMemPrefetchAsync(count, sizeof(int), device_id, stream));

  detail::count_if<<<grid_size, block_size, 0, stream>>>(begin, end, count, p);
  HIPCO_HIP_TRY(hipStreamSynchronize(stream));

  auto res = *count;

  HIPCO_HIP_TRY(hipFree(count));

  return res;
}

template <typename Iterator, typename Predicate>
bool all_of(Iterator begin, Iterator end, Predicate p, hipStream_t stream = 0)
{
  auto const size  = std::distance(begin, end); //end - begin;
  auto const count = count_if(begin, end, p, stream);

  return size == count;
}

template <typename Iterator, typename Predicate>
bool any_of(Iterator begin, Iterator end, Predicate p, hipStream_t stream = 0)
{
  auto const count = count_if(begin, end, p, stream);
  return count > 0;
}

template <typename Iterator, typename Predicate>
bool none_of(Iterator begin, Iterator end, Predicate p, hipStream_t stream = 0)
{
  return not all_of(begin, end, p, stream);
}

template <typename Iterator1, typename Iterator2, typename Predicate>
bool equal(Iterator1 begin1, Iterator1 end1, Iterator2 begin2, Predicate p, hipStream_t stream = 0)
{
  auto const size      =  std::distance(begin1, end1); // end1 - begin1;
  auto const grid_size = (size + block_size - 1) / block_size;

  int* count;
  HIPCO_HIP_TRY(hipMallocManaged(&count, sizeof(int)));

  *count = 0;
  int device_id;
  HIPCO_HIP_TRY(hipGetDevice(&device_id));
  HIPCO_HIP_TRY(hipMemPrefetchAsync(count, sizeof(int), device_id, stream));

  detail::count_if<<<grid_size, block_size, 0, stream>>>(begin1, end1, begin2, count, p);
  HIPCO_HIP_TRY(hipStreamSynchronize(stream));

  auto res = *count;

  HIPCO_HIP_TRY(hipFree(count));

  return res == size;
}

}  // namespace test
}  // namespace hipco
