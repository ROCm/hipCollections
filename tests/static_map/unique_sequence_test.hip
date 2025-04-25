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
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE_SIG("Unique sequence of keys",
                       "",
                       ((typename Key, typename Value, int dummy), Key, Value, dummy), //FIXME(hip): dummy fixes ambiguous get_wrapper calls in catch2
                       (int32_t, int32_t, 1),
                       (int32_t, int64_t, 1),
                       (int64_t, int32_t, 1),
                       (int64_t, int64_t, 1))
{
  constexpr std::size_t num_keys{500'000};
  hipco::static_map<Key, Value> map{
    1'000'000, hipco::empty_key<Key>{-1}, hipco::empty_value<Value>{-1}};

  auto m_view = map.get_device_mutable_view();
  auto view   = map.get_device_view();

  thrust::device_vector<Key> d_keys(num_keys);
  thrust::device_vector<Value> d_values(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  thrust::sequence(thrust::device, d_values.begin(), d_values.end());

  auto pairs_begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator<int>(0),
    [] __device__(int i) /*Todo(HIP): changed auto i to int i */ { return hipco::pair<Key, Value>(i, i); });

  thrust::device_vector<Value> d_results(num_keys);
  thrust::device_vector<bool> d_contained(num_keys);

  // bulk function test cases
  SECTION("All inserted keys-value pairs should be correctly recovered during find")
  {
    map.insert(pairs_begin, pairs_begin + num_keys);
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(d_results.begin(), d_values.begin()));

    REQUIRE(hipco::test::all_of(zip, zip + num_keys, [] __device__(auto const& p) {
      return thrust::get<0>(p) == thrust::get<1>(p);
    }));
  }

  SECTION("All inserted keys-value pairs should be contained")
  {
    map.insert(pairs_begin, pairs_begin + num_keys);
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());

    REQUIRE(hipco::test::all_of(d_contained.begin(), d_contained.end(), thrust::identity{}));
  }

  SECTION("Non-inserted keys-value pairs should not be contained")
  {
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());

    REQUIRE(hipco::test::none_of(d_contained.begin(), d_contained.end(), thrust::identity{}));
  }

  SECTION("Inserting unique keys should return insert success.")
  {
    REQUIRE(hipco::test::all_of(pairs_begin,
                               pairs_begin + num_keys,
                               [m_view] __device__(hipco::pair<Key, Value> const& pair) mutable {
                                 return m_view.insert(pair);
                               }));
  }

  SECTION("Cannot find any key in an empty hash map with non-const view")
  {
    SECTION("non-const view")
    {
      REQUIRE(hipco::test::all_of(pairs_begin,
                                 pairs_begin + num_keys,
                                 [view] __device__(hipco::pair<Key, Value> const& pair) mutable {
                                   return view.find(pair.first) == view.end();
                                 }));
    }
    SECTION("const view")
    {
      REQUIRE(hipco::test::all_of(
        pairs_begin, pairs_begin + num_keys, [view] __device__(hipco::pair<Key, Value> const& pair) {
          return view.find(pair.first) == view.end();
        }));
    }
  }

  SECTION("Keys are all found after inserting many keys.")
  {
    // Bulk insert keys
    thrust::for_each(
      thrust::device,
      pairs_begin,
      pairs_begin + num_keys,
      [m_view] __device__(hipco::pair<Key, Value> const& pair) mutable { m_view.insert(pair); });

    SECTION("non-const view")
    {
      // All keys should be found
      REQUIRE(hipco::test::all_of(pairs_begin,
                                 pairs_begin + num_keys,
                                 [view] __device__(hipco::pair<Key, Value> const& pair) mutable {
                                   auto const found = view.find(pair.first);
                                   return (found != view.end()) and
                                          (found->first.load() == pair.first and
                                           found->second.load() == pair.second);
                                 }));
    }
    SECTION("const view")
    {
      // All keys should be found
      REQUIRE(hipco::test::all_of(
        pairs_begin, pairs_begin + num_keys, [view] __device__(hipco::pair<Key, Value> const& pair) {
          auto const found = view.find(pair.first);
          return (found != view.end()) and
                 (found->first.load() == pair.first and found->second.load() == pair.second);
        }));
    }
  }
}

using size_type = int32_t;

template <typename Map>
__inline__ void test_unique_sequence(Map& map, size_type num_keys)
{
  using Key   = typename Map::key_type;
  using Value = typename Map::mapped_type;

  thrust::device_vector<Key> d_keys(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());

  auto keys_begin = d_keys.begin();
  auto pairs_begin =
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                    [] __device__(auto i) { return hipco::pair<Key, Value>(i, i); });
  thrust::device_vector<bool> d_contained(num_keys);

  auto zip_equal = [] __device__(auto const& p) { return thrust::get<0>(p) == thrust::get<1>(p); };
  auto is_even   = [] __device__(auto const& i) { return i % 2 == 0; };

  SECTION("Non-inserted keys should not be contained.")
  {
    REQUIRE(map.size() == 0);

    map.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(hipco::test::none_of(d_contained.begin(), d_contained.end(), thrust::identity{}));
  }

  SECTION("Non-inserted keys have no matches")
  {
    thrust::device_vector<Value> d_results(num_keys);

    map.find(keys_begin, keys_begin + num_keys, d_results.begin());
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(
      d_results.begin(), thrust::constant_iterator<Key>{map.empty_key_sentinel()}));

    REQUIRE(hipco::test::all_of(zip, zip + num_keys, zip_equal));
  }

  SECTION("All conditionally inserted keys should be contained")
  {
    auto const inserted = map.insert_if(
      pairs_begin, pairs_begin + num_keys, thrust::counting_iterator<std::size_t>(0), is_even);
    REQUIRE(inserted == num_keys / 2);
    REQUIRE(map.size() == num_keys / 2);

    map.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(hipco::test::equal(d_contained.begin(),
                              d_contained.end(),
                              thrust::counting_iterator<std::size_t>(0),
                              [] __device__(auto const& idx_contained, auto const& idx) {
                                return ((idx % 2) == 0) == idx_contained;
                              }));
  }

  map.insert(pairs_begin, pairs_begin + num_keys);
  REQUIRE(map.size() == num_keys);

  SECTION("All inserted keys should be contained.")
  {
    map.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(hipco::test::all_of(d_contained.begin(), d_contained.end(), thrust::identity{}));
  }

  SECTION("Conditional contains should return true on even inputs.")
  {
    map.contains_if(keys_begin,
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
    thrust::device_vector<Value> d_results(num_keys);

    map.find(keys_begin, keys_begin + num_keys, d_results.begin());
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(d_results.begin(), keys_begin));

    REQUIRE(hipco::test::all_of(zip, zip + num_keys, zip_equal));
  }

  SECTION("All inserted key-values should be properly retrieved")
  {
    thrust::device_vector<Value> d_values(num_keys);

    auto const [keys_end, values_end] = map.retrieve_all(keys_begin, d_values.begin());
    REQUIRE(std::distance(keys_begin, keys_end) == num_keys);
    REQUIRE(std::distance(d_values.begin(), values_end) == num_keys);

    thrust::sort(thrust::device, d_values.begin(), values_end);
    REQUIRE(hipco::test::equal(d_values.begin(),
                              values_end,
                              thrust::make_counting_iterator<Value>(0),
                              thrust::equal_to<Value>{}));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "Unique sequence",
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
  constexpr size_type gold_capacity = CGSize == 1 ? 422   // 211 x 1 x 2
                                                  : 412;  // 103 x 2 x 2

  using probe =
    std::conditional_t<Probe == hipco::test::probe_sequence::linear_probing,
                       hipco::experimental::linear_probing<CGSize, hipco::murmurhash3_32<Key>>,
                       hipco::experimental::double_hashing<CGSize,
                                                          hipco::murmurhash3_32<Key>,
                                                          hipco::murmurhash3_32<Key>>>;

  auto map = hipco::experimental::static_map<Key,
                                            Value,
                                            hipco::experimental::extent<size_type>,
                                            cuda::thread_scope_device,
                                            thrust::equal_to<Key>,
                                            probe,
                                            hipco::cuda_allocator<std::byte>,
                                            hipco::experimental::aow_storage<2>>{
    num_keys, hipco::empty_key<Key>{-1}, hipco::empty_value<Value>{-1}};

  REQUIRE(map.capacity() == gold_capacity);

  test_unique_sequence(map, num_keys);
}
