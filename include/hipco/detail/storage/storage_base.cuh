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

#include <cstddef>

namespace hipco {
namespace experimental {
namespace detail {
/**
 * @brief Custom deleter for unique pointer.
 *
 * @tparam SizeType Type of device storage size
 * @tparam Allocator Type of allocator used for device storage
 */
template <typename SizeType, typename Allocator>
struct custom_deleter {
  using pointer = typename Allocator::value_type*;  ///< Value pointer type

  /**
   * @brief Constructor of custom deleter.
   *
   * @param size Number of values to deallocate
   * @param allocator Allocator used for deallocating device storage
   */
  explicit constexpr custom_deleter(SizeType size, Allocator& allocator)
    : size_{size}, allocator_{allocator}
  {
  }

  /**
   * @brief Operator for deallocation
   *
   * @param ptr Pointer to the first value for deallocation
   */
  void operator()(pointer ptr) { allocator_.deallocate(ptr, size_); }

  SizeType size_;         ///< Number of values to delete
  Allocator& allocator_;  ///< Allocator used deallocating values
};

/**
 * @brief Base class of open addressing storage.
 *
 * This class should not be used directly.
 *
 * @tparam Extent Type of extent denoting storage capacity
 */
template <typename Extent>
class storage_base {
 public:
  using extent_type = Extent;                            ///< Storage extent type
  using size_type   = typename extent_type::value_type;  ///< Storage size type

  /**
   * @brief Constructor of base storage.
   *
   * @param size Number of elements to (de)allocate
   */
  __host__ __device__ explicit constexpr storage_base(Extent size) : extent_{size} {}

  /**
   * @brief Gets the total number of elements in the current storage.
   *
   * @return The total number of elements
   */
  [[nodiscard]] __host__ __device__ constexpr size_type capacity() const noexcept
  {
    return static_cast<size_type>(extent_);
  }

  /**
   * @brief Gets the extent of the current storage.
   *
   * @return The extent.
   */
  [[nodiscard]] __host__ __device__ constexpr extent_type extent() const noexcept
  {
    return extent_;
  }

 protected:
  extent_type extent_;  ///< Total number of elements
};

}  // namespace detail
}  // namespace experimental
}  // namespace hipco
