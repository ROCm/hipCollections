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

#pragma once

#include <hipco/detail/bitwise_compare.cuh>

#include <cstddef>

namespace hipco {
namespace experimental {
namespace detail {

/**
 * @brief Enum of equality comparison results.
 */
enum class equal_result : int32_t { UNEQUAL = 0, EMPTY = 1, EQUAL = 2, ERASED = 3 };

/**
 * @brief Key equality wrapper.
 *
 * User-provided equality binary callable cannot be used to compare against sentinel value.
 *
 * @tparam T Right-hand side Element type
 * @tparam Equal Type of user-provided equality binary callable
 */
template <typename T, typename Equal>
struct equal_wrapper {
  // TODO: Clean up the sentinel handling since it's duplicated in ref and equal wrapper
  T empty_sentinel_;   ///< Empty sentinel value
  T erased_sentinel_;  ///< Erased sentinel value
  Equal equal_;        ///< Custom equality callable

  /**
   * @brief Equality wrapper ctor.
   *
   * @param empty_sentinel Empty sentinel value
   * @param erased_sentinel Erased sentinel value
   * @param equal Equality binary callable
   */
  __host__ __device__ constexpr equal_wrapper(T empty_sentinel,
                                              T erased_sentinel,
                                              Equal const& equal) noexcept
    : empty_sentinel_{empty_sentinel}, erased_sentinel_{erased_sentinel}, equal_{equal}
  {
  }

  /**
   * @brief Equality check with the given equality callable.
   *
   * @tparam LHS Left-hand side Element type
   * @tparam RHS Right-hand side Element type
   *
   * @param lhs Left-hand side element to check equality
   * @param rhs Right-hand side element to check equality
   *
   * @return `EQUAL` if `lhs` and `rhs` are equivalent. `UNEQUAL` otherwise.
   */
  template <typename LHS, typename RHS>
  __device__ constexpr equal_result equal_to(LHS const& lhs, RHS const& rhs) const noexcept
  {
    return equal_(lhs, rhs) ? equal_result::EQUAL : equal_result::UNEQUAL;
  }

  /**
   * @brief Order-sensitive equality operator.
   *
   * @note This function always compares the left-hand side element against `empty_sentinel_` value
   * first then perform a equality check with the given `equal_` callable, i.e., `equal_(lhs, rhs)`.
   * @note Container (like set or map) keys MUST be always on the left-hand side.
   *
   * @tparam LHS Left-hand side Element type
   * @tparam RHS Right-hand side Element type
   *
   * @param lhs Left-hand side element to check equality
   * @param rhs Right-hand side element to check equality
   *
   * @return Three way equality comparison result
   */
  template <typename LHS, typename RHS>
  __device__ constexpr equal_result operator()(LHS const& lhs, RHS const& rhs) const noexcept
  {
    return hipco::detail::bitwise_compare(lhs, empty_sentinel_) ? equal_result::EMPTY
                                                               : this->equal_to(lhs, rhs);
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace hipco
