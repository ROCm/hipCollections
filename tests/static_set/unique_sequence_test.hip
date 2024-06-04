/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <hipco/static_set.cuh>

#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <catch2/catch_template_test_macros.hpp>

using size_type = int32_t;

template <typename Set>
__inline__ void test_unique_sequence(Set& set, size_type num_keys)
{
  using Key = typename Set::key_type;

  thrust::device_vector<Key> d_keys(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());

  auto keys_begin = d_keys.begin();
  thrust::device_vector<bool> d_contained(num_keys);

  auto zip_equal = [] __device__(auto const& p) { return thrust::get<0>(p) == thrust::get<1>(p); };
  auto is_even   = [] __device__(auto const& i) { return i % 2 == 0; };

  SECTION("Non-inserted keys should not be contained.")
  {
    REQUIRE(set.size() == 0);

    set.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(hipco::test::none_of(d_contained.begin(), d_contained.end(), thrust::identity{}));
  }

  SECTION("Non-inserted keys have no matches")
  {
    thrust::device_vector<Key> d_results(num_keys);

    set.find(keys_begin, keys_begin + num_keys, d_results.begin());
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(
      d_results.begin(), thrust::constant_iterator<Key>{set.empty_key_sentinel()}));

    REQUIRE(hipco::test::all_of(zip, zip + num_keys, zip_equal));
  }

  SECTION("All conditionally inserted keys should be contained")
  {
    auto const inserted = set.insert_if(
      keys_begin, keys_begin + num_keys, thrust::counting_iterator<std::size_t>(0), is_even);
    REQUIRE(inserted == num_keys / 2);
    REQUIRE(set.size() == num_keys / 2);

    set.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(hipco::test::equal(d_contained.begin(),
                              d_contained.end(),
                              thrust::counting_iterator<std::size_t>(0),
                              [] __device__(auto const& idx_contained, auto const& idx) {
                                return ((idx % 2) == 0) == idx_contained;
                              }));
  }

  set.insert(keys_begin, keys_begin + num_keys);
  REQUIRE(set.size() == num_keys);

  SECTION("All inserted keys should be contained.")
  {
    set.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(hipco::test::all_of(d_contained.begin(), d_contained.end(), thrust::identity{}));
  }

  SECTION("Conditional contains should return true on even inputs.")
  {
    set.contains_if(keys_begin,
                    keys_begin + num_keys,
                    thrust::counting_iterator<std::size_t>(0),
                    is_even,
                    d_contained.begin());
    auto gold_iter =
      thrust::make_transform_iterator(thrust::counting_iterator<std::size_t>(0), is_even);
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(d_contained.begin(), gold_iter));
    REQUIRE(hipco::test::all_of(zip, zip + num_keys, zip_equal));
  }

  SECTION("All inserted keys should be correctly recovered during find")
  {
    thrust::device_vector<Key> d_results(num_keys);

    set.find(keys_begin, keys_begin + num_keys, d_results.begin());
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(d_results.begin(), keys_begin));

    REQUIRE(hipco::test::all_of(zip, zip + num_keys, zip_equal));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "Unique sequence",
  "",
  ((typename Key, hipco::test::probe_sequence Probe, int CGSize), Key, Probe, CGSize),
  (int32_t, hipco::test::probe_sequence::double_hashing, 1),
  (int32_t, hipco::test::probe_sequence::double_hashing, 2),
  (int64_t, hipco::test::probe_sequence::double_hashing, 1),
  (int64_t, hipco::test::probe_sequence::double_hashing, 2),
  (int32_t, hipco::test::probe_sequence::linear_probing, 1),
  (int32_t, hipco::test::probe_sequence::linear_probing, 2),
  (int64_t, hipco::test::probe_sequence::linear_probing, 1),
  (int64_t, hipco::test::probe_sequence::linear_probing, 2))
{
  constexpr size_type num_keys{400};
  constexpr size_type gold_capacity = CGSize == 1 ? 422  // 211 x 1 x 2
                                                  : 412  // 103 x 2 x 2
    ;

  using probe = std::conditional_t<
    Probe == hipco::test::probe_sequence::linear_probing,
    hipco::experimental::linear_probing<CGSize, hipco::default_hash_function<Key>>,
    hipco::experimental::double_hashing<CGSize, hipco::default_hash_function<Key>>>;

  auto set = hipco::experimental::static_set<Key,
                                            hipco::experimental::extent<size_type>,
                                            hip::thread_scope_device,
                                            thrust::equal_to<Key>,
                                            probe,
                                            hipco::cuda_allocator<std::byte>,
                                            hipco::experimental::storage<2>>{
    num_keys, hipco::empty_key<Key>{-1}};

  REQUIRE(set.capacity() == gold_capacity);

  test_unique_sequence(set, num_keys);
}
