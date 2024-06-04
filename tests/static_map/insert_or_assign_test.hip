/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

#include <catch2/catch_template_test_macros.hpp>

using size_type = std::size_t;

template <typename Map>
__inline__ void test_insert_or_assign(Map& map, size_type num_keys)
{
  using Key   = typename Map::key_type;
  using Value = typename Map::mapped_type;

  // Insert pairs
  auto pairs_begin =
    thrust::make_transform_iterator(thrust::counting_iterator<size_type>(0),
                                    [] __device__(auto i) { return hipco::pair<Key, Value>(i, i); });

  auto const initial_size = map.insert(pairs_begin, pairs_begin + num_keys);
  REQUIRE(initial_size == num_keys);  // all keys should be inserted

  // Query pairs have the same keys but different payloads
  auto query_pairs_begin = thrust::make_transform_iterator(
    thrust::counting_iterator<size_type>(0),
    [] __device__(auto i) { return hipco::pair<Key, Value>(i, i * 2); });

  map.insert_or_assign(query_pairs_begin, query_pairs_begin + num_keys);

  auto const updated_size = map.size();
  // all keys are present in the map so the size shouldn't change
  REQUIRE(updated_size == initial_size);

  thrust::device_vector<Key> d_keys(num_keys);
  thrust::device_vector<Key> d_values(num_keys);
  map.retrieve_all(d_keys.begin(), d_values.begin());

  auto gold_values_begin = thrust::make_transform_iterator(thrust::counting_iterator<size_type>(0),
                                                           [] __device__(auto i) { return i * 2; });

  thrust::sort(thrust::device, d_values.begin(), d_values.end());
  REQUIRE(hipco::test::equal(
    d_values.begin(), d_values.end(), gold_values_begin, thrust::equal_to<Value>{}));
}

TEMPLATE_TEST_CASE_SIG(
  "Insert or assign",
  "",
  ((typename Key, typename Value, hipco::test::probe_sequence Probe, int CGSize),
   Key,
   Value,
   Probe,
   CGSize),
  (int32_t, int32_t, hipco::test::probe_sequence::double_hashing, 1),
  (int32_t, int64_t, hipco::test::probe_sequence::double_hashing, 1),
  (int32_t, int32_t, hipco::test::probe_sequence::double_hashing, 2),
  (int32_t, int64_t, hipco::test::probe_sequence::double_hashing, 2),
  (int64_t, int32_t, hipco::test::probe_sequence::double_hashing, 1),
  (int64_t, int64_t, hipco::test::probe_sequence::double_hashing, 1),
  (int64_t, int32_t, hipco::test::probe_sequence::double_hashing, 2),
  (int64_t, int64_t, hipco::test::probe_sequence::double_hashing, 2),
  (int32_t, int32_t, hipco::test::probe_sequence::linear_probing, 1),
  (int32_t, int64_t, hipco::test::probe_sequence::linear_probing, 1),
  (int32_t, int32_t, hipco::test::probe_sequence::linear_probing, 2),
  (int32_t, int64_t, hipco::test::probe_sequence::linear_probing, 2),
  (int64_t, int32_t, hipco::test::probe_sequence::linear_probing, 1),
  (int64_t, int64_t, hipco::test::probe_sequence::linear_probing, 1),
  (int64_t, int32_t, hipco::test::probe_sequence::linear_probing, 2),
  (int64_t, int64_t, hipco::test::probe_sequence::linear_probing, 2))
{
  constexpr size_type num_keys{400};

  using probe =
    std::conditional_t<Probe == hipco::test::probe_sequence::linear_probing,
                       hipco::experimental::linear_probing<CGSize, hipco::murmurhash3_32<Key>>,
                       hipco::experimental::double_hashing<CGSize,
                                                          hipco::murmurhash3_32<Key>,
                                                          hipco::murmurhash3_32<Key>>>;

  auto map = hipco::experimental::static_map<Key,
                                            Value,
                                            hipco::experimental::extent<size_type>,
                                            hip::thread_scope_device,
                                            thrust::equal_to<Key>,
                                            probe,
                                            hipco::cuda_allocator<std::byte>,
                                            hipco::experimental::storage<2>>{
    num_keys, hipco::empty_key<Key>{-1}, hipco::empty_value<Value>{-1}};

  test_insert_or_assign(map, num_keys);
}
