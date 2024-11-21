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

#include <hipco/utility/fast_int.cuh>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstdint>
#include <type_traits>

TEMPLATE_TEST_CASE(
  "utility::fast_int tests", "", std::int32_t, std::uint32_t, std::int64_t, std::uint64_t)
{
  TestType value           = GENERATE(1, 2, 9, 32, 4123, 8192, 4312456);
  TestType lhs             = GENERATE(1, 2, 9, 32, 4123, 8192, 4312456);
  constexpr auto max_value = std::numeric_limits<TestType>::max();

  hipco::utility::fast_int fast_value{value};

  SECTION("Should be explicitly convertible to the underlying integer type.")
  {
    REQUIRE(static_cast<TestType>(fast_value) == value);
  }

  SECTION("Fast div/mod should produce correct result.")
  {
    INFO(lhs << " /% " << value);
    REQUIRE(lhs / fast_value == lhs / value);
    REQUIRE(lhs % fast_value == lhs % value);
  }

  SECTION("Fast div/mod with maximum rhs value should produce correct result.")
  {
    INFO(lhs << " /% " << max_value);
    hipco::utility::fast_int fast_max{max_value};
    REQUIRE(lhs / fast_max == lhs / max_value);
    REQUIRE(lhs % fast_max == lhs % max_value);
  }

  SECTION("Fast div/mod with maximum lhs value should produce correct result.")
  {
    INFO(max_value << " /% " << value);
    REQUIRE(max_value / fast_value == max_value / value);
    REQUIRE(max_value % fast_value == max_value % value);
  }
}
