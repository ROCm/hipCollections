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

#include <hipco/static_map.cuh>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Static map capacity", "")
{
  using Key        = int32_t;
  using T          = int32_t;
  using ProbeT     = hipco::experimental::double_hashing<1, hipco::default_hash_function<Key>>;
  using Equal      = thrust::equal_to<Key>;
  using AllocatorT = hipco::cuda_allocator<std::byte>;
  using StorageT   = hipco::experimental::storage<2>;

  SECTION("zero capacity is allowed.")
  {
    auto constexpr gold_capacity = 4;

    using extent_type = hipco::experimental::extent<std::size_t, 0>;
    hipco::experimental::static_map<Key,
                                   T,
                                   extent_type,
                                   hip::thread_scope_device,
                                   Equal,
                                   ProbeT,
                                   AllocatorT,
                                   StorageT>
      map{extent_type{}, hipco::empty_key<Key>{-1}, hipco::empty_value<T>{-1}};
    auto const capacity = map.capacity();
    REQUIRE(capacity == gold_capacity);

    auto ref                = map.ref(hipco::experimental::insert);
    auto const ref_capacity = ref.capacity();
    REQUIRE(ref_capacity == gold_capacity);
  }

  SECTION("negative capacity (ikr -_-||) is also allowed.")
  {
    auto constexpr gold_capacity = 4;

    using extent_type = hipco::experimental::extent<int32_t>;
    hipco::experimental::static_map<Key,
                                   T,
                                   extent_type,
                                   hip::thread_scope_device,
                                   Equal,
                                   ProbeT,
                                   AllocatorT,
                                   StorageT>
      map{extent_type{-10}, hipco::empty_key<Key>{-1}, hipco::empty_value<T>{-1}};
    auto const capacity = map.capacity();
    REQUIRE(capacity == gold_capacity);

    auto ref                = map.ref(hipco::experimental::insert);
    auto const ref_capacity = ref.capacity();
    REQUIRE(ref_capacity == gold_capacity);
  }

  constexpr std::size_t num_keys{400};

  SECTION("Dynamic extent is evaluated at run time.")
  {
    auto constexpr gold_capacity = 422;  // 211 x 2

    using extent_type = hipco::experimental::extent<std::size_t>;
    hipco::experimental::static_map<Key,
                                   T,
                                   extent_type,
                                   hip::thread_scope_device,
                                   Equal,
                                   ProbeT,
                                   AllocatorT,
                                   StorageT>
      map{num_keys, hipco::empty_key<Key>{-1}, hipco::empty_value<T>{-1}};
    auto const capacity = map.capacity();
    REQUIRE(capacity == gold_capacity);

    auto ref                = map.ref(hipco::experimental::insert);
    auto const ref_capacity = ref.capacity();
    REQUIRE(ref_capacity == gold_capacity);
  }

  SECTION("map can be constructed from plain integer.")
  {
    auto constexpr gold_capacity = 422;  // 211 x 2

    hipco::experimental::static_map<Key,
                                   T,
                                   std::size_t,
                                   hip::thread_scope_device,
                                   Equal,
                                   ProbeT,
                                   AllocatorT,
                                   StorageT>
      map{num_keys, hipco::empty_key<Key>{-1}, hipco::empty_value<T>{-1}};
    auto const capacity = map.capacity();
    REQUIRE(capacity == gold_capacity);

    auto ref                = map.ref(hipco::experimental::insert);
    auto const ref_capacity = ref.capacity();
    REQUIRE(ref_capacity == gold_capacity);
  }

  SECTION("map can be constructed from plain integer and load factor.")
  {
    auto constexpr gold_capacity = 502;  // 251 x 2

    hipco::experimental::static_map<Key,
                                   T,
                                   std::size_t,
                                   hip::thread_scope_device,
                                   Equal,
                                   ProbeT,
                                   AllocatorT,
                                   StorageT>
      map{num_keys, 0.8, hipco::empty_key<Key>{-1}, hipco::empty_value<T>{-1}};
    auto const capacity = map.capacity();
    REQUIRE(capacity == gold_capacity);

    auto ref                = map.ref(hipco::experimental::insert);
    auto const ref_capacity = ref.capacity();
    REQUIRE(ref_capacity == gold_capacity);
  }

  SECTION("Dynamic extent is evaluated at run time.")
  {
    auto constexpr gold_capacity = 412;  // 103 x 2 x 2

    using probe = hipco::experimental::linear_probing<2, hipco::default_hash_function<Key>>;
    auto map    = hipco::experimental::static_map<Key,
                                              T,
                                              hipco::experimental::extent<std::size_t>,
                                              hip::thread_scope_device,
                                              Equal,
                                              probe,
                                              AllocatorT,
                                              StorageT>{
      num_keys, hipco::empty_key<Key>{-1}, hipco::empty_value<T>{-1}};

    auto const capacity = map.capacity();
    REQUIRE(capacity == gold_capacity);

    auto ref                = map.ref(hipco::experimental::insert);
    auto const ref_capacity = ref.capacity();
    REQUIRE(ref_capacity == gold_capacity);
  }
}
