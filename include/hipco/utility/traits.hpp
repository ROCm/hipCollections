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

#include <thrust/device_reference.h>
#include <thrust/tuple.h>

#include <type_traits>

namespace hipco {

/**
 * @brief Customization point that can be specialized to indicate that it is safe to perform bitwise
 * equality comparisons on the object-representation of objects of type `T`.
 *
 * By default, only types where `std::has_unique_object_representations_v<T>` is true are safe for
 * bitwise equality. However, this can be too restrictive for some types, e.g., floating point
 * types.
 *
 * User-defined specializations of `is_bitwise_comparable` are allowed, but it is the users
 * responsibility to ensure values do not occur that would lead to unexpected behavior. For example,
 * if a `NaN` bit pattern were used as the empty sentinel value, it may not compare bitwise equal to
 * other `NaN` bit patterns.
 *
 */
template <typename T, typename = void>
struct is_bitwise_comparable : std::false_type {
};

/// By default, only types with unique object representations are allowed
template <typename T>
struct is_bitwise_comparable<T, std::enable_if_t<std::has_unique_object_representations_v<T>>>
  : std::true_type {
};

template <typename T>
inline constexpr bool is_bitwise_comparable_v = is_bitwise_comparable<T>::value;

/**
 * @brief Declares that a type `Type` is bitwise comparable.
 *
 */
#define HIPCO_DECLARE_BITWISE_COMPARABLE(Type)           \
  namespace hipco {                                      \
  template <>                                           \
  struct is_bitwise_comparable<Type> : std::true_type { \
  };                                                    \
  }

template <bool value, typename... Args>
inline constexpr bool dependent_bool_value = value;

template <typename... Args>
inline constexpr bool dependent_false = dependent_bool_value<false, Args...>;

}  // namespace hipco
