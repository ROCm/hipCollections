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

namespace hipco {
namespace experimental {
inline namespace op {
// TODO enum class of int32_t instead of struct
// https://github.com/NVIDIA/cuCollections/issues/239
/**
 * @brief `insert` operator tag
 */
struct insert_tag {
} inline constexpr insert;

/**
 * @brief `insert_and_find` operator tag
 */
struct insert_and_find_tag {
} inline constexpr insert_and_find;

/**
 * @brief `insert_or_assign` operator tag
 */
struct insert_or_assign_tag {
} inline constexpr insert_or_assign;

/**
 * @brief `erase` operator tag
 */
struct erase_tag {
} inline constexpr erase;

/**
 * @brief `contains` operator tag
 */
struct contains_tag {
} inline constexpr contains;

/**
 * @brief `find` operator tag
 */
struct find_tag {
} inline constexpr find;

}  // namespace op
}  // namespace experimental
}  // namespace hipco

#include <hipco/detail/operator.inl>
