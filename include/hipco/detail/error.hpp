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

#include <hipco/utility/error.hpp>

#include <hip/hip_runtime_api.h>

#define STRINGIFY_DETAIL(x) #x
#define HIPCO_STRINGIFY(x)   STRINGIFY_DETAIL(x)

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call. If the call does not return
 * `hipSuccess`, invokes hipGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 *
 * Defaults to throwing `hipco::cuda_error`, but a custom exception may also be
 * specified.
 *
 * Example:
 * ```c++
 *
 * // Throws `hipco::cuda_error` if `hipMalloc` fails
 * HIPCO_HIP_TRY(hipMalloc(&p, 100));
 *
 * // Throws `std::runtime_error` if `hipMalloc` fails
 * HIPCO_HIP_TRY(hipMalloc(&p, 100), std::runtime_error);
 * ```
 *
 */
#define HIPCO_HIP_TRY(...)                                               \
  GET_HIPCO_HIP_TRY_MACRO(__VA_ARGS__, HIPCO_HIP_TRY_2, HIPCO_HIP_TRY_1) \
  (__VA_ARGS__)
#define GET_HIPCO_HIP_TRY_MACRO(_1, _2, NAME, ...) NAME
#define HIPCO_HIP_TRY_2(_call, _exception_type)                                                    \
  do {                                                                                             \
    hipError_t error = (_call);                                                             \
    if (hipSuccess != error) {                                                                    \
      error = hipGetLastError();                                                                          \
      throw _exception_type{std::string{"CUDA error at: "} + __FILE__ + HIPCO_STRINGIFY(__LINE__) + \
                            ": " + hipGetErrorName(error) + " " + hipGetErrorString(error)};     \
    }                                                                                              \
  } while (0);
#define HIPCO_HIP_TRY_1(_call) HIPCO_HIP_TRY_2(_call, hipco::cuda_error)

/**
 * @brief Error checking macro for CUDA runtime API that asserts the result is
 * equal to `hipSuccess`.
 *
 */
#define HIPCO_ASSERT_CUDA_SUCCESS(expr) \
  do {                                 \
    hipError_t const status = (expr); \
    assert(hipSuccess == status);     \
  } while (0)

/**
 * @brief Macro for checking (pre-)conditions that throws an exception when
 * a condition is violated.
 *
 * Defaults to throwing `hipco::logic_error`, but a custom exception may also be
 * specified.
 *
 * Example usage:
 * ```
 * // throws hipco::logic_error
 * HIPCO_EXPECTS(p != nullptr, "Unexpected null pointer");
 *
 * // throws std::runtime_error
 * HIPCO_EXPECTS(p != nullptr, "Unexpected nullptr", std::runtime_error);
 * ```
 * @param ... This macro accepts either two or three arguments:
 *   - The first argument must be an expression that evaluates to true or
 *     false, and is the condition being checked.
 *   - The second argument is a string literal used to construct the `what` of
 *     the exception.
 *   - When given, the third argument is the exception to be thrown. When not
 *     specified, defaults to `hipco::logic_error`.
 * @throw `_exception_type` if the condition evaluates to 0 (false).
 */
#define HIPCO_EXPECTS(...)                                             \
  GET_HIPCO_EXPECTS_MACRO(__VA_ARGS__, HIPCO_EXPECTS_3, HIPCO_EXPECTS_2) \
  (__VA_ARGS__)

#define GET_HIPCO_EXPECTS_MACRO(_1, _2, _3, NAME, ...) NAME

#define HIPCO_EXPECTS_3(_condition, _reason, _exception_type)                    \
  do {                                                                          \
    static_assert(std::is_base_of_v<std::exception, _exception_type>);          \
    (_condition) ? static_cast<void>(0)                                         \
                 : throw _exception_type /*NOLINT(bugprone-macro-parentheses)*/ \
      {"HIPCO failure at: " __FILE__ ":" HIPCO_STRINGIFY(__LINE__) ": " _reason}; \
  } while (0)

#define HIPCO_EXPECTS_2(_condition, _reason) HIPCO_EXPECTS_3(_condition, _reason, hipco::logic_error)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * Example usage:
 * ```c++
 * // Throws `hipco::logic_error`
 * HIPCO_FAIL("Unsupported code path");
 *
 * // Throws `std::runtime_error`
 * HIPCO_FAIL("Unsupported code path", std::runtime_error);
 * ```
 *
 * @param ... This macro accepts either one or two arguments:
 *   - The first argument is a string literal used to construct the `what` of
 *     the exception.
 *   - When given, the second argument is the exception to be thrown. When not
 *     specified, defaults to `hipco::logic_error`.
 * @throw `_exception_type` if the condition evaluates to 0 (false).
 */
#define HIPCO_FAIL(...)                                       \
  GET_HIPCO_FAIL_MACRO(__VA_ARGS__, HIPCO_FAIL_2, HIPCO_FAIL_1) \
  (__VA_ARGS__)

#define GET_HIPCO_FAIL_MACRO(_1, _2, NAME, ...) NAME

#define HIPCO_FAIL_2(_what, _exception_type)      \
  /*NOLINTNEXTLINE(bugprone-macro-parentheses)*/ \
  throw _exception_type { "HIPCO failure at:" __FILE__ ":" HIPCO_STRINGIFY(__LINE__) ": " _what }

#define HIPCO_FAIL_1(_what) HIPCO_FAIL_2(_what, hipco::logic_error)
