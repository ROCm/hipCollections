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
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include <catch2/catch_template_test_macros.hpp>

#include <tuple>

// insert key type
template <typename T>
struct alignas(8) key_pair {
  T a;
  T b;

  __host__ __device__ key_pair() {}
  __host__ __device__ key_pair(T x) : a{x}, b{x} {}

  // Device equality operator is mandatory due to libcudacxx bug:
  // https://github.com/NVIDIA/libcudacxx/issues/223
  __device__ bool operator==(key_pair const& other) const { return a == other.a and b == other.b; }
};

// probe key type
template <typename T>
struct alignas(8) key_triplet {
  T a;
  T b;
  T c;

  __host__ __device__ key_triplet() {}
  __host__ __device__ key_triplet(T x) : a{x}, b{x}, c{x} {}

  // Device equality operator is mandatory due to libcudacxx bug:
  // https://github.com/NVIDIA/libcudacxx/issues/223
  __device__ bool operator==(key_triplet const& other) const
  {
    return a == other.a and b == other.b and c == other.c;
  }
};

// User-defined device hasher
struct alignas(8) custom_hasher {
  template <typename CustomKey>
  __host__ __device__ uint32_t operator()(CustomKey const& k)//todo(HIP): __host__ needed to select the right template, compiler issue?
  {
    return thrust::raw_reference_cast(k).a;
  };
};

// User-defined device key equality
struct alignas(8) custom_key_equal {
  template <typename LHS, typename RHS>
  __host__ __device__ bool operator()(LHS const& lhs, RHS const& rhs) //todo(HIP): __host__ needed to select the right template, compiler issue?
  {
    return thrust::raw_reference_cast(lhs).a == thrust::raw_reference_cast(rhs).a;
  }
};

TEMPLATE_TEST_CASE("Heterogeneous lookup",
                   "",
#if defined(HIPCO_HAS_INDEPENDENT_THREADS)  // Key type larger than 8B only supported for sm_70 and
                                           // up
                   int64_t,
#endif
                   int32_t)
{
  using Key      = key_pair<TestType>;
  using Value    = TestType;
  using ProbeKey = key_triplet<TestType>;

  auto const sentinel_key   = Key{-1};
  auto const sentinel_value = Value{-1};

  constexpr std::size_t num      = 100;
  constexpr std::size_t capacity = num * 2;
  hipco::static_map<Key, Value> map{
    capacity, hipco::empty_key<Key>{sentinel_key}, hipco::empty_value<Value>{sentinel_value}};

  auto insert_pairs = thrust::make_transform_iterator(
    thrust::counting_iterator<int>(0),
    [] __device__(int i) { return hipco::pair<Key, Value>(i, i); });
  auto probe_keys = thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                    [] __device__(int i) { return ProbeKey(i); });

  SECTION("All inserted keys-value pairs should be contained")
  {
    thrust::device_vector<bool> contained(num);
    map.insert(insert_pairs, insert_pairs + num, custom_hasher{}, custom_key_equal{});
    map.contains(
      probe_keys, probe_keys + num, contained.begin(), custom_hasher{}, custom_key_equal{});
    REQUIRE(hipco::test::all_of(contained.begin(), contained.end(), thrust::identity{}));
  }

  SECTION("Non-inserted keys-value pairs should not be contained")
  {
    thrust::device_vector<bool> contained(num);
    map.contains(
      probe_keys, probe_keys + num, contained.begin(), custom_hasher{}, custom_key_equal{});
    REQUIRE(hipco::test::none_of(contained.begin(), contained.end(), thrust::identity{}));
  }
}
