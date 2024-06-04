/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <hipco/detail/probe_sequence_impl.cuh>

namespace hipco {

/**
 * @brief Public linear probing scheme class.
 *
 * Linear probing is efficient when few collisions are present. Performance hints:
 * - Use linear probing when collisions are rare. e.g. low occupancy or low multiplicity.
 * - `CGSize` = 1 or 2 when hash map is small (10'000'000 or less), 4 or 8 otherwise.
 *
 * `Hash` should be callable object type.
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 * @tparam Hash Unary callable type
 */
template <uint32_t CGSize, typename Hash>
class linear_probing : public detail::probe_sequence_base<CGSize> {
 public:
  using probe_sequence_base_type =
    detail::probe_sequence_base<CGSize>;  ///< The base probe scheme type
  using probe_sequence_base_type::cg_size;
  using probe_sequence_base_type::vector_width;

  /// Type of implementation details
  template <typename Key, typename Value, hip::thread_scope Scope>
  using impl = detail::linear_probing_impl<Key, Value, Scope, vector_width(), CGSize, Hash>;
};

/**
 *
 * @brief Public double hashing scheme class.
 *
 * Default probe sequence for `hipco::static_multimap`. Double hashing shows superior
 * performance when dealing with high multiplicty and/or high occupancy use cases. Performance
 * hints:
 * - `CGSize` = 1 or 2 when hash map is small (10'000'000 or less), 4 or 8 otherwise.
 *
 * `Hash1` and `Hash2` should be callable object type.
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 * @tparam Hash1 Unary callable type
 * @tparam Hash2 Unary callable type
 */
template <uint32_t CGSize, typename Hash1, typename Hash2 = Hash1>
class double_hashing : public detail::probe_sequence_base<CGSize> {
 public:
  using probe_sequence_base_type =
    detail::probe_sequence_base<CGSize>;  ///< The base probe scheme type
  using probe_sequence_base_type::cg_size;
  using probe_sequence_base_type::vector_width;

  /// Type of implementation details
  template <typename Key, typename Value, hip::thread_scope Scope>
  using impl = detail::double_hashing_impl<Key, Value, Scope, vector_width(), CGSize, Hash1, Hash2>;
};

/**
 *
 * @brief Coalesced probing class (CAUTION: to be only used for benchmarking purposes!) with
 * coalesced groups.
 *
 * Probe sequence that generates a coalesced memory access pattern for benchmarking purposes.
 * This does not realize a hash function and ignores keys. It should not be used for anything
 * else than experimenting with performance (random memory access vs coalesced memory access).
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 */
template <uint32_t CGSize>
class coalesced_probing : public detail::probe_sequence_base<CGSize> {
 public:
  using probe_sequence_base_type =
    detail::probe_sequence_base<CGSize>;  ///< The base probe scheme type
  using probe_sequence_base_type::cg_size;
  using probe_sequence_base_type::vector_width;

  /// Type of implementation details
  template <typename Key, typename Value, hip::thread_scope Scope>
  using impl = detail::coalesced_probing_impl<Key, Value, Scope, vector_width(), CGSize>;
};

}  // namespace hipco
