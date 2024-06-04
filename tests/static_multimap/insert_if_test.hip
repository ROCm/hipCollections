/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <hipco/static_multimap.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <catch2/catch_template_test_macros.hpp>

template <typename Key, typename Map, typename PairIt, typename KeyIt>
__inline__ void test_insert_if(Map& map, PairIt pair_begin, KeyIt key_begin, std::size_t size)
{
  // 50% insertion
  auto pred_lambda = [] __device__(Key k) { return k % 2 == 0; };

  map.insert_if(pair_begin, pair_begin + size, key_begin, pred_lambda);

  auto res = map.get_size();
  REQUIRE(res * 2 == size);

  auto num = map.count(key_begin, key_begin + size);
  REQUIRE(num * 2 == size);
}

TEMPLATE_TEST_CASE_SIG(
  "Tests of insert_if",
  "",
  ((typename Key, typename Value, hipco::test::probe_sequence Probe), Key, Value, Probe),
  (int32_t, int32_t, hipco::test::probe_sequence::linear_probing),
  (int32_t, int64_t, hipco::test::probe_sequence::linear_probing),
  (int64_t, int64_t, hipco::test::probe_sequence::linear_probing),
  (int32_t, int32_t, hipco::test::probe_sequence::double_hashing),
  (int32_t, int64_t, hipco::test::probe_sequence::double_hashing),
  (int64_t, int64_t, hipco::test::probe_sequence::double_hashing))
{
  constexpr std::size_t num_keys{1'000};

  thrust::device_vector<Key> d_keys(num_keys);
  thrust::device_vector<hipco::pair<Key, Value>> d_pairs(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  // multiplicity = 1
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_keys),
                    d_pairs.begin(),
                    [] __device__(auto i) {
                      return hipco::pair<Key, Value>{i, i};
                    });

  if constexpr (Probe == hipco::test::probe_sequence::linear_probing) {
    hipco::static_multimap<Key,
                          Value,
                          hip::thread_scope_device,
                          hipco::cuda_allocator<char>,
                          hipco::linear_probing<1, hipco::murmurhash3_32<Key>>>
      map{num_keys * 2, hipco::empty_key<Key>{-1}, hipco::empty_value<Value>{-1}};
    test_insert_if<Key>(map, d_pairs.begin(), d_keys.begin(), num_keys);
  }
  if constexpr (Probe == hipco::test::probe_sequence::double_hashing) {
    hipco::static_multimap<Key, Value> map{
      num_keys * 2, hipco::empty_key<Key>{-1}, hipco::empty_value<Value>{-1}};
    test_insert_if<Key>(map, d_pairs.begin(), d_keys.begin(), num_keys);
  }
}
