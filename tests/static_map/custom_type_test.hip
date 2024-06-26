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
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include <catch2/catch_template_test_macros.hpp>

#include <tuple>

// User-defined key type
template <typename T>
struct alignas(8) key_pair_type { //no hip support for unaligned atomics SWDEV-393058
  T a;
  T b;

  __host__ __device__ key_pair_type() {}
  __host__ __device__ key_pair_type(T x) : a{x}, b{x} {}

  // Device equality operator is mandatory due to libcudacxx bug:
  // https://github.com/NVIDIA/libcudacxx/issues/223
  __device__ bool operator==(key_pair_type const& other) const
  {
    return a == other.a and b == other.b;
  }
};

// User-defined key type
template <typename T>
struct alignas(8) large_key_type { //no hip support for unaligned atomics SWDEV-393058
  T a;
  T b;
  T c;

  __host__ __device__ large_key_type() {}
  __host__ __device__ large_key_type(T x) : a{x}, b{x}, c{x} {}

  // Device equality operator is mandatory due to libcudacxx bug:
  // https://github.com/NVIDIA/libcudacxx/issues/223
  __device__ bool operator==(large_key_type const& other) const
  {
    return a == other.a and b == other.b and c == other.c;
  }
};

// User-defined value type
template <typename T>
struct alignas(8) value_pair_type { //no hip support for unaligned atomics SWDEV-393058
  T f;
  T s;

  __host__ __device__ value_pair_type() {}
  __host__ __device__ value_pair_type(T x) : f{x}, s{x} {}

  __device__ bool operator==(value_pair_type const& other) const
  {
    return f == other.f and s == other.s;
  }
};

// User-defined device hasher
struct hash_custom_key {
  template <typename custom_type>
  __host__ __device__ uint32_t operator()(custom_type k) //todo(HIP): __host__ needed to select the right template, compiler issue?
  {
    return thrust::raw_reference_cast(k).a;
  };
};

// User-defined device key equality
struct custom_key_equals {
  template <typename lhs_type, typename rhs_type>
  __host__ __device__ bool operator()(lhs_type lhs, rhs_type rhs)  //todo(HIP): __host__ needed to select the right template, compiler issue?
  {
    return lhs == static_cast<lhs_type>(rhs);
  }
};

TEMPLATE_TEST_CASE_SIG("User defined key and value type",
                       "",
                       ((typename Key, typename Value, int dummy), Key, Value, dummy),
#if defined(HIPCO_HAS_INDEPENDENT_THREADS)  // Key type larger than 8B only supported for sm_70 and
                                           // up
                       (key_pair_type<int64_t>, value_pair_type<int32_t>, 1),
                       (key_pair_type<int64_t>, value_pair_type<int64_t>, 1),
                       (large_key_type<int32_t>, value_pair_type<int32_t>, 1),
#endif
                       (key_pair_type<int32_t>, value_pair_type<int32_t>, 1))
{
  auto const sentinel_key   = Key{-1};
  auto const sentinel_value = Value{-1};

  constexpr std::size_t num      = 100;
  constexpr std::size_t capacity = num * 2;
  hipco::static_map<Key, Value> map{
    capacity, hipco::empty_key<Key>{sentinel_key}, hipco::empty_value<Value>{sentinel_value}};

  thrust::device_vector<Key> insert_keys(num);
  thrust::device_vector<Value> insert_values(num);

  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num),
                    insert_keys.begin(),
                    [] __device__(auto i) { return Key{i}; });

  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num),
                    insert_values.begin(),
                    [] __device__(auto i) { return Value{i}; });

  auto insert_pairs = thrust::make_transform_iterator(
    thrust::make_counting_iterator<int>(0),
    [] __host__ __device__(int i) { return hipco::pair<Key, Value>(i, i); }); //todo(hip): auto keyword not working with rocthrust for parameter i, double check if __host__ is needed

  SECTION("All inserted keys-value pairs should be correctly recovered during find")
  {
    thrust::device_vector<Value> found_values(num);
    map.insert(insert_pairs, insert_pairs + num, hash_custom_key{}, custom_key_equals{});

    REQUIRE(num == map.get_size());

    map.find(insert_keys.begin(),
             insert_keys.end(),
             found_values.begin(),
             hash_custom_key{},
             custom_key_equals{});

    REQUIRE(hipco::test::equal(insert_values.begin(),
                              insert_values.end(),
                              found_values.begin(),
                              [] __device__(Value lhs, Value rhs) {
                                return std::tie(lhs.f, lhs.s) == std::tie(rhs.f, rhs.s);
                              }));
  }

  SECTION("All inserted keys-value pairs should be contained")
  {
    thrust::device_vector<bool> contained(num);
    map.insert(insert_pairs, insert_pairs + num, hash_custom_key{}, custom_key_equals{});
    map.contains(insert_keys.begin(),
                 insert_keys.end(),
                 contained.begin(),
                 hash_custom_key{},
                 custom_key_equals{});
    REQUIRE(hipco::test::all_of(contained.begin(), contained.end(), thrust::identity{}));
  }

  SECTION("All conditionally inserted keys-value pairs should be contained")
  {
    thrust::device_vector<bool> contained(num);
    map.insert_if(
      insert_pairs,
      insert_pairs + num,
      thrust::counting_iterator<int>(0),
      [] __device__(auto const& key) { return (key % 2) == 0; },
      hash_custom_key{},
      custom_key_equals{});

    REQUIRE(num / 2 == map.get_size());

    map.contains(insert_keys.begin(),
                 insert_keys.end(),
                 contained.begin(),
                 hash_custom_key{},
                 custom_key_equals{});

    REQUIRE(hipco::test::equal(contained.begin(),
                              contained.end(),
                              thrust::counting_iterator<int>(0),
                              [] __device__(auto const& idx_contained, auto const& idx) {
                                return ((idx % 2) == 0) == idx_contained;
                              }));
  }

  SECTION("Non-inserted keys-value pairs should not be contained")
  {
    thrust::device_vector<bool> contained(num);
    map.contains(insert_keys.begin(),
                 insert_keys.end(),
                 contained.begin(),
                 hash_custom_key{},
                 custom_key_equals{});
    REQUIRE(hipco::test::none_of(contained.begin(), contained.end(), thrust::identity{}));
  }

  SECTION("All inserted keys-value pairs should be contained")
  {
    thrust::device_vector<bool> contained(num);
    map.insert(insert_pairs, insert_pairs + num, hash_custom_key{}, custom_key_equals{});
    auto view = map.get_device_view();
    REQUIRE(hipco::test::all_of(
      insert_pairs, insert_pairs + num, [view] __device__(hipco::pair<Key, Value> const& pair) {
        return view.contains(pair.first, hash_custom_key{}, custom_key_equals{});
      }));
  }

  SECTION("Inserting unique keys should return insert success.")
  {
    auto m_view = map.get_device_mutable_view();
    REQUIRE(
      hipco::test::all_of(insert_pairs,
                         insert_pairs + num,
                         [m_view] __device__(hipco::pair<Key, Value> const& pair) mutable {
                           return m_view.insert(pair, hash_custom_key{}, custom_key_equals{});
                         }));
  }

  SECTION("Cannot find any key in an empty hash map")
  {
    SECTION("non-const view")
    {
      auto view = map.get_device_view();
      REQUIRE(hipco::test::all_of(
        insert_pairs,
        insert_pairs + num,
        [view] __device__(hipco::pair<Key, Value> const& pair) mutable {
          return view.find(pair.first, hash_custom_key{}, custom_key_equals{}) == view.end();
        }));
    }

    SECTION("const view")
    {
      auto const view = map.get_device_view();
      REQUIRE(hipco::test::all_of(
        insert_pairs,
        insert_pairs + num,
        [view] __device__(hipco::pair<Key, Value> const& pair) {
          return view.find(pair.first, hash_custom_key{}, custom_key_equals{}) == view.end();
        }));
    }
  }
}
