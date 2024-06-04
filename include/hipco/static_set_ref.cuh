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

#include <hipco/detail/open_addressing/open_addressing_ref_impl.cuh>
#include <hipco/hash_functions.cuh>
#include <hipco/operator.hpp>
#include <hipco/probing_scheme.cuh>
#include <hipco/sentinel.cuh>
#include <hipco/storage.cuh>

#include <hip/std/atomic>

#include <memory>

namespace hipco {
namespace experimental {

/**
 * @brief Device non-owning "ref" type that can be used in device code to perform arbitrary
 * operations defined in `include/hipco/operator.hpp`
 *
 * @note Concurrent modify and lookup will be supported if both kinds of operators are specified
 * during the ref construction.
 * @note hipCollections data stuctures always place the slot keys on the left-hand
 * side when invoking the key comparison predicate.
 * @note Ref types are trivially-copyable and are intended to be passed by value.
 * @note `ProbingScheme::cg_size` indicates how many threads are used to handle one independent
 * device operation. `cg_size == 1` uses the scalar (or non-CG) code paths.
 *
 * @throw If the size of the given key type is larger than 8 bytes
 * @throw If the given key type doesn't have unique object representations, i.e.,
 * `hipco::bitwise_comparable_v<Key> == false`
 * @throw If the probing scheme type is not inherited from `hipco::detail::probing_scheme_base`
 *
 * @tparam Key Type used for keys. Requires `hipco::is_bitwise_comparable_v<Key>` returning true
 * @tparam Scope The scope in which operations will be performed by individual threads.
 * @tparam KeyEqual Binary callable type used to compare two keys for equality
 * @tparam ProbingScheme Probing scheme (see `include/hipco/probing_scheme.cuh` for options)
 * @tparam StorageRef Storage ref type
 * @tparam Operators Device operator options defined in `include/hipco/operator.hpp`
 */
template <typename Key,
          hip::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class static_set_ref
  : public detail::operator_impl<
      Operators,
      static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>>... {
  using impl_type =
    detail::open_addressing_ref_impl<Key, Scope, KeyEqual, ProbingScheme, StorageRef>;

 public:
  using key_type            = Key;                                     ///< Key Type
  using probing_scheme_type = ProbingScheme;                           ///< Type of probing scheme
  using storage_ref_type    = StorageRef;                              ///< Type of storage ref
  using window_type         = typename storage_ref_type::window_type;  ///< Window type
  using value_type          = typename storage_ref_type::value_type;   ///< Storage element type
  using extent_type         = typename storage_ref_type::extent_type;  ///< Extent type
  using size_type           = typename storage_ref_type::size_type;    ///< Probing scheme size type
  using key_equal           = KeyEqual;  ///< Type of key equality binary callable
  using iterator            = typename storage_ref_type::iterator;   ///< Slot iterator type
  using const_iterator = typename storage_ref_type::const_iterator;  ///< Const slot iterator type

  static constexpr auto cg_size = probing_scheme_type::cg_size;  ///< Cooperative group size
  static constexpr auto window_size =
    storage_ref_type::window_size;  ///< Number of elements handled per window

  /**
   * @brief Constructs static_set_ref.
   *
   * @param empty_key_sentinel Sentinel indicating empty key
   * @param predicate Key equality binary callable
   * @param probing_scheme Probing scheme
   * @param storage_ref Non-owning ref of slot storage
   */
  __host__ __device__ explicit constexpr static_set_ref(hipco::empty_key<Key> empty_key_sentinel,
                                                        KeyEqual const& predicate,
                                                        ProbingScheme const& probing_scheme,
                                                        StorageRef storage_ref) noexcept;

  /**
   * @brief Constructs static_set_ref.
   *
   * @param empty_key_sentinel Sentinel indicating empty key
   * @param erased_key_sentinel Sentinel indicating erased key
   * @param predicate Key equality binary callable
   * @param probing_scheme Probing scheme
   * @param storage_ref Non-owning ref of slot storage
   */
  __host__ __device__ explicit constexpr static_set_ref(hipco::empty_key<Key> empty_key_sentinel,
                                                        hipco::erased_key<Key> erased_key_sentinel,
                                                        KeyEqual const& predicate,
                                                        ProbingScheme const& probing_scheme,
                                                        StorageRef storage_ref) noexcept;

  /**
   * @brief Operator-agnostic move constructor.
   *
   * @tparam OtherOperators Operator set of the `other` object
   *
   * @param other Object to construct `*this` from
   */
  template <typename... OtherOperators>
  __host__ __device__ explicit constexpr static_set_ref(
    static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, OtherOperators...>&&
      other) noexcept;

  /**
   * @brief Gets the maximum number of elements the container can hold.
   *
   * @return The maximum number of elements the container can hold
   */
  [[nodiscard]] __host__ __device__ constexpr auto capacity() const noexcept;

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  [[nodiscard]] __host__ __device__ constexpr key_type empty_key_sentinel() const noexcept;

  /**
   * @brief Creates a reference with new operators from the current object.
   *
   * Note that this function uses move semantics and thus invalidates the current object.
   *
   * @warning Using two or more reference objects to the same container but with
   * a different operator set at the same time results in undefined behavior.
   *
   * @tparam NewOperators List of `hipco::op::*_tag` types
   *
   * @param ops List of operators, e.g., `hipco::insert`
   *
   * @return `*this` with `NewOperators...`
   */
  template <typename... NewOperators>
  [[nodiscard]] __host__ __device__ auto with(NewOperators... ops) && noexcept;

 private:
  impl_type impl_;

  // Mixins need to be friends with this class in order to access private members
  template <typename Op, typename Ref>
  friend class detail::operator_impl;

  // Refs with other operator sets need to be friends too
  template <typename Key_,
            hip::thread_scope Scope_,
            typename KeyEqual_,
            typename ProbingScheme_,
            typename StorageRef_,
            typename... Operators_>
  friend class static_set_ref;
};

}  // namespace experimental
}  // namespace hipco

#include <hipco/detail/static_set/static_set_ref.inl>
