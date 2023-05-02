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

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <cuco/detail/error.hpp>

namespace cuco {
/**
 * @brief A device allocator using `hipMalloc`/`hipFree` to satisfy (de)allocations.
 *
 * @tparam T The allocator's value type
 */
template <typename T>
class cuda_allocator {
 public:
  using value_type = T;  ///< Allocator's value type

  cuda_allocator() = default;

  /**
   * @brief Copy constructor.
   */
  template <class U>
  cuda_allocator(cuda_allocator<U> const&) noexcept
  {
  }

  /**
   * @brief Allocates storage for `n` objects of type `T` using `hipMalloc`.
   *
   * @param n The number of objects to allocate storage for
   * @return Pointer to the allocated storage
   */
  value_type* allocate(std::size_t n)
  {
    value_type* p;
    CUCO_CUDA_TRY(hipMalloc(&p, sizeof(value_type) * n));
    return p;
  }

  /**
   * @brief Deallocates storage pointed to by `p`.
   *
   * @param p Pointer to memory to deallocate
   */
  void deallocate(value_type* p, std::size_t) { CUCO_CUDA_TRY(hipFree(p)); }
};

/**
 * @brief Equality comparison operator.
 *
 * @tparam T Value type of LHS object
 * @tparam U Value type of RHS object
 *
 * @return `true` iff given arguments are equal
 */
template <typename T, typename U>
bool operator==(cuda_allocator<T> const&, cuda_allocator<U> const&) noexcept
{
  return true;
}

/**
 * @brief Inequality comparison operator.
 *
 * @tparam T Value type of LHS object
 * @tparam U Value type of RHS object
 *
 * @param lhs Left-hand side object to compare
 * @param rhs Right-hand side object to compare
 *
 * @return `true` iff given arguments are not equal
 */
template <typename T, typename U>
bool operator!=(cuda_allocator<T> const& lhs, cuda_allocator<U> const& rhs) noexcept
{
  return not(lhs == rhs);
}

}  // namespace cuco
