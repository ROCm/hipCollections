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

#include <hipco/utility/traits.hpp>

#include <type_traits>

namespace hipco {
namespace experimental {
namespace detail {

/**
 * @brief CRTP mixin which augments a given `Reference` with an `Operator`.
 *
 * @throw If the operator is not defined in `include/hipco/operator.hpp`
 *
 * @tparam Operator Operator type, i.e., `hipco::op::*_tag`
 * @tparam Reference The reference type.
 *
 * @note This primary template should never be instantiated.
 */
template <typename Operator, typename Reference>
class operator_impl {
  static_assert(hipco::dependent_false<Operator, Reference>,
                "Operator type is not supported by reference type.");
};

/**
 * @brief Checks if the given `Operator` is contained in a list of `Operators`.
 *
 * @tparam Operator Operator type, i.e., `hipco::op::*_tag`
 * @tparam Operators List of operators to search in
 *
 * @return `true` if `Operator` is contained in `Operators`, `false` otherwise.
 */
template <typename Operator, typename... Operators>
static constexpr bool has_operator()
{
  return ((std::is_same_v<Operators, Operator>) || ...);
}

}  // namespace detail
}  // namespace experimental
}  // namespace hipco
