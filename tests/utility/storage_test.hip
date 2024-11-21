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

#include <hipco/pair.cuh>
#include <hipco/aow_storage.cuh>
#include <hipco/extent.cuh>
#include <hipco/utility/allocator.hpp>

#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE_SIG("Storage tests",
                       "",
                       ((typename Key, typename Value, int dummy), Key, Value, dummy),
                       (int32_t, int32_t, 1),
                       (int32_t, int64_t, 1),
                       (int64_t, int64_t, 1))
{
  constexpr std::size_t size{1'000};
  constexpr int window_size{2};
  constexpr std::size_t gold_capacity{2'000};

  using allocator_type = hipco::cuda_allocator<char>;
  auto allocator       = allocator_type{};

  SECTION("Allocate array of pairs with AoS storage.")
  {
    auto s =
      hipco::experimental::aow_storage<hipco::pair<Key, Value>,
                                      window_size,
                                      hipco::experimental::extent<std::size_t>,
                                      allocator_type>(hipco::experimental::extent{size}, allocator);
    auto const num_windows = s.num_windows();
    auto const capacity    = s.capacity();

    REQUIRE(num_windows == size);
    REQUIRE(capacity == gold_capacity);
  }

  SECTION("Allocate array of pairs with AoS storage with static extent.")
  {
    using extent_type = hipco::experimental::extent<std::size_t, size>;
    auto s            = hipco::experimental::
      aow_storage<hipco::pair<Key, Value>, window_size, extent_type, allocator_type>(extent_type{},
                                                                                    allocator);
    auto const num_windows = s.num_windows();
    auto const capacity    = s.capacity();

    STATIC_REQUIRE(num_windows == size);
    STATIC_REQUIRE(capacity == gold_capacity);
  }

  SECTION("Allocate array of keys with AoS storage.")
  {
    auto s = hipco::experimental::
      aow_storage<Key, window_size, hipco::experimental::extent<std::size_t>, allocator_type>(
        hipco::experimental::extent{size}, allocator);
    auto const num_windows = s.num_windows();
    auto const capacity    = s.capacity();

    REQUIRE(num_windows == size);
    REQUIRE(capacity == gold_capacity);
  }

  SECTION("Allocate array of keys with AoS storage with static extent.")
  {
    using extent_type = hipco::experimental::extent<std::size_t, size>;
    auto s = hipco::experimental::aow_storage<Key, window_size, extent_type, allocator_type>(
      extent_type{}, allocator);
    auto const num_windows = s.num_windows();
    auto const capacity    = s.capacity();

    STATIC_REQUIRE(num_windows == size);
    STATIC_REQUIRE(capacity == gold_capacity);
  }
}
