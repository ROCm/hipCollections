/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <hip/std/atomic>  // cuda::thread_scope

namespace cuco {

/**
 * @brief Strongly-typed wrapper for `cuda::thread_scope`.
 *
 * @tparam Scope `cuda::thread_scope` to be wrapped
 */
template <cuda::thread_scope Scope>
struct cuda_thread_scope {
  /**
   * @brief Implicit conversion to `cuda::thread_scope`.
   *
   * @return The wrapped `cuda::thread_scope`
   */
  __host__ __device__ constexpr operator cuda::thread_scope() const noexcept { return Scope; }
};

// alias definitions
inline constexpr auto thread_scope_system = cuda_thread_scope<cuda::thread_scope_system>{};
inline constexpr auto thread_scope_device = cuda_thread_scope<cuda::thread_scope_device>{};
inline constexpr auto thread_scope_block  = cuda_thread_scope<cuda::thread_scope_block>{};
inline constexpr auto thread_scope_thread = cuda_thread_scope<cuda::thread_scope_thread>{};

}  // namespace cuco
