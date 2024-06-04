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

#include <hipco/detail/storage/storage.cuh>

namespace hipco {
namespace experimental {

/**
 * @brief Public storage class.
 *
 * @note This is a public interface used to control storage window size. A window consists of a
 * number of contiguous slots. The window size defines the workload granularity for each CUDA
 * thread, i.e., how many slots a thread would concurrently operate on when performing modify or
 * lookup operations. hipCollections uses the AoW storage to supersede the raw flat slot storage due
 * to its superior granularity control: When window size equals one, AoW performs the same as the
 * flat storage. If the underlying operation is more memory bandwidth bound, e.g., high occupancy
 * multimap operations, a larger window size can reduce the length of probing sequences thus improve
 * runtime performance.
 *
 * @tparam WindowSize Number of elements per window storage
 */
template <int32_t WindowSize>
class storage {
 public:
  /// Number of slots per window storage
  static constexpr int32_t window_size = WindowSize;

  /// Type of implementation details
  template <class T, class Extent, class Allocator>
  using impl = aow_storage<T, window_size, Extent, Allocator>;
};

}  // namespace experimental
}  // namespace hipco
