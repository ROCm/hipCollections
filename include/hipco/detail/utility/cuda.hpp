/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <hipco/detail/utility/math.hpp>

namespace hipco {
namespace detail {

using index_type = int64_t;  ///< CUDA thread index type

/// Default block size
constexpr int32_t default_block_size() noexcept { return 128; }
/// Default stride
constexpr int32_t default_stride() noexcept { return 1; }

/**
 * @brief Computes the desired 1D grid size with the given parameters
 *
 * @param num Number of elements to handle in the kernel
 * @param cg_size Number of threads per CUDA Cooperative Group
 * @param stride Number of elements to be handled by each thread
 * @param block_size Number of threads in each thread block
 *
 * @return The resulting grid size
 */
constexpr auto grid_size(index_type num,
                         int32_t cg_size    = 1,
                         int32_t stride     = default_stride(),
                         int32_t block_size = default_block_size()) noexcept
{
  return int_div_ceil(cg_size * num, stride * block_size);
}

}  // namespace detail
}  // namespace hipco
