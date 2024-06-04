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

#pragma once

#include <hipco/detail/storage/storage_base.cuh>

#include <hip/std/array>

#include <cstddef>
#include <cstdint>

namespace hipco {
namespace experimental {
namespace detail {
/**
￼ * @brief Window data structure type
￼ *
￼ * @tparam T Window slot type
￼ * @tparam WindowSize Number of elements per window
￼ */
template <typename T, int32_t WindowSize>
struct window : public hip::std::array<T, WindowSize> {
 public:
  static int32_t constexpr window_size = WindowSize;  ///< Number of slots per window
};

/**
 * @brief Base class of array of slot windows open addressing storage.
 *
 * @note This should NOT be used directly.
 *
 * @tparam T Slot type
 * @tparam WindowSize Number of slots in each window
 * @tparam Extent Type of extent denoting the number of windows
 */
template <typename T, int32_t WindowSize, typename Extent>
class aow_storage_base : public storage_base<Extent> {
 public:
  /**
   * @brief The number of elements (slots) processed per window.
   */
  static constexpr int32_t window_size = WindowSize;

  using extent_type = typename storage_base<Extent>::extent_type;  ///< Storage extent type
  using size_type   = typename storage_base<Extent>::size_type;    ///< Storage size type

  using value_type  = T;                                ///< Slot type
  using window_type = window<value_type, window_size>;  ///< Slot window type

  /**
   * @brief Constructor of AoW base storage.
   *
   * @param size Number of windows to store
   */
  __host__ __device__ explicit constexpr aow_storage_base(Extent size) : storage_base<Extent>{size}
  {
  }

  /**
   * @brief Gets the total number of slot windows in the current storage.
   *
   * @return The total number of slot windows
   */
  [[nodiscard]] __host__ __device__ constexpr size_type num_windows() const noexcept
  {
    return storage_base<Extent>::capacity();
  }

  /**
   * @brief Gets the total number of slots in the current storage.
   *
   * @return The total number of slots
   */
  [[nodiscard]] __host__ __device__ constexpr size_type capacity() const noexcept
  {
    return storage_base<Extent>::capacity() * window_size;
  }

  /**
   * @brief Gets the window extent of the current storage.
   *
   * @return The window extent.
   */
  [[nodiscard]] __host__ __device__ constexpr extent_type window_extent() const noexcept
  {
    return storage_base<Extent>::extent();
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace hipco
