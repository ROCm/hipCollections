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

#include <hipco/static_set.cuh>

#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <catch2/catch_template_test_macros.hpp>

template <typename Set>
__inline__ void test_unique_sequence(Set& set, std::size_t num_keys)
{
  using Key = typename Set::key_type;

  thrust::device_vector<Key> d_keys(num_keys);
  thrust::sequence(d_keys.begin(), d_keys.end());
  auto keys_begin = d_keys.begin();

  SECTION("Non-inserted keys should not be contained.")
  {
    REQUIRE(set.size() == 0);

    auto keys_end = set.retrieve_all(keys_begin);
    REQUIRE(std::distance(keys_begin, keys_end) == 0);
  }

  set.insert(keys_begin, keys_begin + num_keys);
  REQUIRE(set.size() == num_keys);

  SECTION("All inserted key/value pairs should be contained.")
  {
    thrust::device_vector<Key> d_res(num_keys);
    auto d_res_end = set.retrieve_all(d_res.begin());
    thrust::sort(d_res.begin(), d_res_end);
    REQUIRE(hipco::test::equal(
      d_res.begin(), d_res_end, thrust::counting_iterator<Key>(0), thrust::equal_to<Key>{}));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "Retrieve all",
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
  constexpr std::size_t num_keys{400};
  auto constexpr gold_capacity = CGSize == 1 ? 409  // 409 x 1 x 1
                                             : 422  // 211 x 2 x 1
    ;

  using probe = std::conditional_t<
    Probe == hipco::test::probe_sequence::linear_probing,
    hipco::experimental::linear_probing<CGSize, hipco::default_hash_function<Key>>,
    hipco::experimental::double_hashing<CGSize, hipco::default_hash_function<Key>>>;

  auto set = hipco::experimental::static_set<Key,
                                            hipco::experimental::extent<std::size_t>,
                                            hip::thread_scope_device,
                                            thrust::equal_to<Key>,
                                            probe,
                                            hipco::cuda_allocator<std::byte>,
                                            hipco::experimental::storage<1>>{
    num_keys, hipco::empty_key<Key>{-1}};

  REQUIRE(set.capacity() == gold_capacity);

  test_unique_sequence(set, num_keys);
}
