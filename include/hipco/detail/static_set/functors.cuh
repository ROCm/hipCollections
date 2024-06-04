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

namespace hipco {
namespace experimental {
namespace static_set_ns {
namespace detail {

/**
 * @brief Device functor returning whether the input slot indexed by `idx` is filled.
 *
 * @tparam T The slot content type
 */
template <typename T>
struct slot_is_filled {
  T empty_sentinel_;   ///< The value of the empty key sentinel
  T erased_sentinel_;  ///< Key value that represents an erased slot

  /**
   * @brief Constructs `slot_is_filled` functor with the given sentinels.
   *
   * @param empty_sentinel Sentinel indicating empty slot
   * @param erased_sentinel Sentinel indicating erased slot
   */
  explicit constexpr slot_is_filled(T const& empty_sentinel, T const& erased_sentinel) noexcept
    : empty_sentinel_{empty_sentinel}, erased_sentinel_{erased_sentinel}
  {
  }

  /**
   * @brief Indicates if the target slot `slot` is filled.
   *
   * @tparam T Slot content type
   *
   * @param slot The slot
   *
   * @return `true` if slot is filled
   */
  __host__ __device__ constexpr bool operator()(T const& slot) const noexcept //todo(HIP): double check if __host__ is needed
  {
    return not(hipco::detail::bitwise_compare(empty_sentinel_, slot) or
               hipco::detail::bitwise_compare(erased_sentinel_, slot));
  }
};

}  // namespace detail
}  // namespace static_set_ns
}  // namespace experimental
}  // namespace hipco
