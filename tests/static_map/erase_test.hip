/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE_SIG("erase key", "", ((typename T, int dummy), T, dummy), (int32_t, 1), (int64_t, 1)) //FIXME(hip): dummy fixes ambiguous get_wrapper calls in catch2
{
  using Key   = T;
  using Value = T;

  constexpr std::size_t num_keys = 1'000'000;
  constexpr std::size_t capacity = 1'100'000;

  hipco::static_map<Key, Value> map{
    capacity, hipco::empty_key<Key>{-1}, hipco::empty_value<Value>{-1}, hipco::erased_key<Key>{-2}};

  thrust::device_vector<Key> d_keys(num_keys);
  thrust::device_vector<Value> d_values(num_keys);
  thrust::device_vector<bool> d_keys_exist(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end(), 1);
  thrust::sequence(thrust::device, d_values.begin(), d_values.end(), 1);

  auto pairs_begin =
    thrust::make_zip_iterator(thrust::make_tuple(d_keys.begin(), d_values.begin()));

  SECTION("Check basic insert/erase")
  {
    map.insert(pairs_begin, pairs_begin + num_keys);

    REQUIRE(map.get_size() == num_keys);

    map.erase(d_keys.begin(), d_keys.end());

    REQUIRE(map.get_size() == 0);

    map.contains(d_keys.begin(), d_keys.end(), d_keys_exist.begin());

    REQUIRE(hipco::test::none_of(d_keys_exist.begin(),
                                d_keys_exist.end(),
                                [] __device__(const bool key_found) { return key_found; }));

    map.insert(pairs_begin, pairs_begin + num_keys);

    REQUIRE(map.get_size() == num_keys);

    map.contains(d_keys.begin(), d_keys.end(), d_keys_exist.begin());

    REQUIRE(hipco::test::all_of(d_keys_exist.begin(),
                               d_keys_exist.end(),
                               [] __device__(const bool key_found) { return key_found; }));

    map.erase(d_keys.begin(), d_keys.begin() + num_keys / 2);
    map.contains(d_keys.begin(), d_keys.end(), d_keys_exist.begin());

    REQUIRE(hipco::test::none_of(d_keys_exist.begin(),
                                d_keys_exist.begin() + num_keys / 2,
                                [] __device__(const bool key_found) { return key_found; }));

    REQUIRE(hipco::test::all_of(d_keys_exist.begin() + num_keys / 2,
                               d_keys_exist.end(),
                               [] __device__(const bool key_found) { return key_found; }));

    map.erase(d_keys.begin() + num_keys / 2, d_keys.end());
    REQUIRE(map.get_size() == 0);
  }
}
