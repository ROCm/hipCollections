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

#include <utils.hpp>

#include <hipco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <catch2/catch_template_test_macros.hpp>

#define SIZE 10
__device__ int A[SIZE];

template <typename T>
struct custom_equals {
  __device__ bool operator()(T lhs, T rhs) { return A[lhs] == A[rhs]; }
};

TEMPLATE_TEST_CASE_SIG(
  "Key comparison against sentinel", "", ((typename T, int dummy), T, dummy), (int32_t, 1), (int64_t, 1))
{
  using Key   = T;
  using Value = T;

  constexpr std::size_t num_keys{SIZE};
  hipco::static_map<Key, Value> map{
    SIZE * 2, hipco::empty_key<Key>{-1}, hipco::empty_value<Value>{-1}};

  auto m_view = map.get_device_mutable_view();
  auto view   = map.get_device_view();

  int h_A[SIZE];
  for (int i = 0; i < SIZE; i++) {
    h_A[i] = i;
  }
  HIPCO_HIP_TRY(hipMemcpyToSymbol(A, h_A, SIZE * sizeof(int)));

  auto pairs_begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator<T>(0),
    [] __device__(auto i) { return hipco::pair<Key, Value>(i, i); });

  SECTION(
    "Tests of non-CG insert: The custom `key_equal` can never be used to compare against sentinel")
  {
    REQUIRE(hipco::test::all_of(
      pairs_begin,
      pairs_begin + num_keys,
      [m_view] __device__(hipco::pair<Key, Value> const& pair) mutable {
        return m_view.insert(pair, hipco::default_hash_function<Key>{}, custom_equals<Key>{});
      }));
  }

  SECTION(
    "Tests of CG insert: The custom `key_equal` can never be used to compare against sentinel")
  {
    map.insert(
      pairs_begin, pairs_begin + num_keys,hipco::default_hash_function<Key>{}, custom_equals<Key>{});
    // All keys inserted via custom `key_equal` should be found
    REQUIRE(hipco::test::all_of(pairs_begin, 
                              pairs_begin + num_keys, 
                              [view] __device__(hipco::pair<Key, Value> const& pair) {
        auto const found = view.find(pair.first);
        return (found != view.end()) and
               (found->first.load() == pair.first and 
               found->second.load() == pair.second);
      }));
  }
}
