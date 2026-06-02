/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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

// NOTE(HIP/AMD): `rank` lives in its own header so that dynamic_bitset kernels
// can use the concrete type without including dynamic_bitset.cuh, which would
// re-introduce the dynamic_bitset.cuh -> dynamic_bitset.inl -> kernels.cuh
// include cycle and break header self-containment.

#pragma once

#include <cuda/std/array>
#include <cuda/std/cstdint>

#include <climits>

namespace cuco {
namespace experimental {
namespace detail {

/**
 * @brief Struct to store ranks of bits at 256-bit intervals (or blocks)
 *
 * This struct encodes a list of four rank values using base + offset format
 * e.g. [1000, 1005, 1006, 1009] is stored as base = 1000, offsets = [5, 6, 9]
 * base uses 40 bits, split between one uint32_t and one uint8_t
 * each offset uses 8 bits
 */
struct rank {
  uint32_t base_hi_;                      ///< Upper 32 bits of base
  uint8_t base_lo_;                       ///< Lower 8 bits of base
  cuda::std::array<uint8_t, 3> offsets_;  ///< Offsets for 64-bit sub-intervals, relative to base

  /**
   * @brief Gets base rank of current 256-bit interval
   *
   * @return The base rank
   */
  __host__ __device__ constexpr uint64_t base() const noexcept
  {
    return (static_cast<uint64_t>(base_hi_) << CHAR_BIT) | base_lo_;
  }

  /**
   * @brief Sets base rank of current 256-bit interval
   *
   * @param base Base rank
   */
  __host__ __device__ constexpr void set_base(uint64_t base) noexcept
  {
    base_hi_ = static_cast<uint32_t>(base >> CHAR_BIT);
    base_lo_ = static_cast<uint8_t>(base);
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
