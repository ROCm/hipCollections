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
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <catch2/catch_template_test_macros.hpp>

// Custom pair equal
template <typename Key, typename Value>
struct pair_equal {
  __device__ bool operator()(const hipco::pair<Key, Value>& lhs,
                             const hipco::pair<Key, Value>& rhs) const
  {
    return lhs.first == rhs.first;
  }
};

template <typename Key, typename Value, typename Map, typename PairIt>
__inline__ void test_pair_functions(Map& map, PairIt pair_begin, std::size_t num_pairs)
{
  map.insert(pair_begin, pair_begin + num_pairs);
  HIPCO_HIP_TRY(hipStreamSynchronize(0));

  auto res = map.get_size();
  REQUIRE(res == num_pairs);

  // query pair matching rate = 50%
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_pairs),
                    pair_begin,
                    [] __device__(auto i) {
                      return hipco::pair<Key, Value>{i, i};
                    });

  SECTION("pair_contains returns true for all inserted pairs and false for non-inserted ones.")
  {
    thrust::device_vector<bool> result(num_pairs);
    auto res_begin = result.begin();
    map.pair_contains(pair_begin, pair_begin + num_pairs, res_begin, pair_equal<Key, Value>{});

    auto true_iter  = thrust::make_constant_iterator(true);
    auto false_iter = thrust::make_constant_iterator(false);

    REQUIRE(
      hipco::test::equal(res_begin, res_begin + num_pairs / 2, true_iter, thrust::equal_to<bool>{}));
    REQUIRE(hipco::test::equal(
      res_begin + num_pairs / 2, res_begin + num_pairs, false_iter, thrust::equal_to<bool>{}));
  }

  SECTION("Output of pair_count and pair_retrieve should be coherent.")
  {
    auto num = map.pair_count(pair_begin, pair_begin + num_pairs, pair_equal<Key, Value>{});

    auto out1_begin = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));
    auto out2_begin = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));

    REQUIRE(num == num_pairs);

    auto [out1_end, out2_end] = map.pair_retrieve(
      pair_begin, pair_begin + num_pairs, out1_begin, out2_begin, pair_equal<Key, Value>{});
    std::size_t const size = std::distance(out2_begin, out1_end);

    REQUIRE(size == num_pairs);
  }

  SECTION("Output of pair_count_outer and pair_retrieve_outer should be coherent.")
  {
    auto num = map.pair_count_outer(pair_begin, pair_begin + num_pairs, pair_equal<Key, Value>{});

    auto out1_begin = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));
    auto out2_begin = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));

    REQUIRE(num == (num_pairs + num_pairs / 2));

    auto [out1_end, out2_end] = map.pair_retrieve_outer(
      pair_begin, pair_begin + num_pairs, out1_begin, out2_begin, pair_equal<Key, Value>{});
    std::size_t const size = std::distance(out1_begin, out1_end);

    REQUIRE(size == (num_pairs + num_pairs / 2));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "Tests of pair functions",
  "",
  ((typename Key, typename Value, hipco::test::probe_sequence Probe), Key, Value, Probe),
  (int32_t, int32_t, hipco::test::probe_sequence::linear_probing),
  (int32_t, int64_t, hipco::test::probe_sequence::linear_probing),
  (int64_t, int64_t, hipco::test::probe_sequence::linear_probing),
  (int32_t, int32_t, hipco::test::probe_sequence::double_hashing),
  (int32_t, int64_t, hipco::test::probe_sequence::double_hashing),
  (int64_t, int64_t, hipco::test::probe_sequence::double_hashing))
{
  constexpr std::size_t num_pairs{4};
  thrust::device_vector<hipco::pair<Key, Value>> d_pairs(num_pairs);

  // pair multiplicity = 2
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_pairs),
                    d_pairs.begin(),
                    [] __device__(auto i) {
                      return hipco::pair<Key, Value>{i / 2, i};
                    });

  if constexpr (Probe == hipco::test::probe_sequence::linear_probing) {
    hipco::static_multimap<Key,
                          Value,
                          hip::thread_scope_device,
                          hipco::cuda_allocator<char>,
                          hipco::linear_probing<1, hipco::murmurhash3_32<Key>>>
      map{num_pairs * 2, hipco::empty_key<Key>{-1}, hipco::empty_value<Value>{-1}};
    test_pair_functions<Key, Value>(map, d_pairs.begin(), num_pairs);
  }
  if constexpr (Probe == hipco::test::probe_sequence::double_hashing) {
    hipco::static_multimap<Key, Value> map{
      num_pairs * 2, hipco::empty_key<Key>{-1}, hipco::empty_value<Value>{-1}};
    test_pair_functions<Key, Value>(map, d_pairs.begin(), num_pairs);
  }
}
