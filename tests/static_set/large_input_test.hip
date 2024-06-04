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
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <catch2/catch_template_test_macros.hpp>

template <typename Set>
__inline__ void test_unique_sequence(Set& set, bool* res_begin, std::size_t num_keys)
{
  using Key = typename Set::key_type;

  auto const keys_begin = thrust::counting_iterator<Key>(0);
  auto const keys_end   = thrust::counting_iterator<Key>(num_keys);

  SECTION("Non-inserted keys should not be contained.")
  {
    REQUIRE(set.size() == 0);

    set.contains(keys_begin, keys_end, res_begin);
    REQUIRE(hipco::test::none_of(res_begin, res_begin + num_keys, thrust::identity{}));
  }

  set.insert(keys_begin, keys_end);
  REQUIRE(set.size() == num_keys);

  SECTION("All inserted key/value pairs should be contained.")
  {
    set.contains(keys_begin, keys_end, res_begin);
    REQUIRE(hipco::test::all_of(res_begin, res_begin + num_keys, thrust::identity{}));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "Large input",
  "",
  ((typename Key, hipco::test::probe_sequence Probe, int CGSize), Key, Probe, CGSize),
  (int32_t, hipco::test::probe_sequence::double_hashing, 1),
  (int32_t, hipco::test::probe_sequence::double_hashing, 2),
  (int64_t, hipco::test::probe_sequence::double_hashing, 1),
  (int64_t, hipco::test::probe_sequence::double_hashing, 2))
{
  constexpr std::size_t num_keys{1'200'000'000};

  using extent_type = hipco::experimental::extent<std::size_t>;
  using probe       = hipco::experimental::double_hashing<CGSize, hipco::default_hash_function<Key>>;

  try {
    auto set = hipco::experimental::
      static_set<Key, extent_type, hip::thread_scope_device, thrust::equal_to<Key>, probe>{
        num_keys * 2, hipco::empty_key<Key>{-1}};

    thrust::device_vector<bool> d_contained(num_keys);
    test_unique_sequence(set, d_contained.data().get(), num_keys);
  } catch (hipco::cuda_error&) {
    SKIP("Out of memory");
  } catch (std::bad_alloc&) {
    SKIP("Out of memory");
  }
}
