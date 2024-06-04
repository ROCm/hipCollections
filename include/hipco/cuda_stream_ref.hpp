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

#include <hip/hip_runtime.h>

#include <cstddef>

namespace hipco {
namespace experimental {

/**
 * @brief Strongly-typed non-owning wrapper for CUDA streams with default constructor.
 *
 * This wrapper is simply a "view": it does not own the lifetime of the stream it wraps.
 */
class cuda_stream_ref {
 public:
  constexpr cuda_stream_ref()                       = default;  ///< Default constructor
  constexpr cuda_stream_ref(cuda_stream_ref const&) = default;  ///< Copy constructor
  constexpr cuda_stream_ref(cuda_stream_ref&&)      = default;  ///< Move constructor

  /**
   * @brief Copy-assignment operator.
   *
   * @return Copy of this stream reference.
   */
  constexpr cuda_stream_ref& operator=(cuda_stream_ref const&) = default;

  /**
   * @brief Move-assignment operator.
   *
   * @return New location of this stream reference.
   */
  constexpr cuda_stream_ref& operator=(cuda_stream_ref&&) = default;  ///< Move-assignment operator

  ~cuda_stream_ref() = default;

  constexpr cuda_stream_ref(int)            = delete;  //< Prevent cast from literal 0
  constexpr cuda_stream_ref(std::nullptr_t) = delete;  //< Prevent cast from nullptr

  /**
   * @brief Implicit conversion from `hipStream_t`.
   *
   * @param stream The CUDA stream to reference.
   */
  constexpr cuda_stream_ref(hipStream_t stream) noexcept : stream_{stream} {}

  /**
   * @brief Get the wrapped stream.
   *
   * @return The wrapped stream.
   */
  [[nodiscard]] constexpr hipStream_t value() const noexcept { return stream_; }

  /**
   * @brief Implicit conversion to `hipStream_t`.
   *
   * @return The underlying `hipStream_t`.
   */
  constexpr operator hipStream_t() const noexcept { return value(); }

  /**
   * @brief Return true if the wrapped stream is the CUDA per-thread default stream.
   *
   * @return True if the wrapped stream is the per-thread default stream; else false.
   */
  [[nodiscard]] inline bool is_per_thread_default() const noexcept;

  /**
   * @brief Return true if the wrapped stream is explicitly the CUDA legacy default stream.
   *
   * @return True if the wrapped stream is the default stream; else false.
   */
  [[nodiscard]] inline bool is_default() const noexcept;

  /**
   * @brief Synchronize the viewed CUDA stream.
   *
   * Calls `hipStreamSynchronize()`.
   *
   * @throw hipco::cuda_error if stream synchronization fails
   */
  void synchronize() const;

 private:
  hipStream_t stream_{};
};

/**
 * @brief Static `cuda_stream_ref` of the default stream (stream 0), for convenience
 */
static constexpr cuda_stream_ref cuda_stream_default{};

/**
 * @brief Static `cuda_stream_ref` of cudaStreamLegacy, for convenience
 */
//TODO(HIP): is there an equivalent?
static const cuda_stream_ref cuda_stream_legacy{};

/**
 * @brief Static `cuda_stream_ref` of hipStreamPerThread, for convenience
 */
static const cuda_stream_ref cuda_stream_per_thread{hipStreamPerThread};

// /**
//  * @brief Equality comparison operator for streams
//  *
//  * @param lhs The first stream view to compare
//  * @param rhs The second stream view to compare
//  * @return true if equal, false if unequal
//  */
// inline bool operator==(cuda_stream_ref lhs, cuda_stream_ref rhs)
// {
//   return lhs.value() == rhs.value();
// }

// /**
//  * @brief Inequality comparison operator for streams
//  *
//  * @param lhs The first stream view to compare
//  * @param rhs The second stream view to compare
//  * @return true if unequal, false if equal
//  */
// inline bool operator!=(cuda_stream_ref lhs, cuda_stream_ref rhs) { return not(lhs == rhs); }

}  // namespace experimental
}  // namespace hipco

#include <hipco/detail/cuda_stream_ref.inl>