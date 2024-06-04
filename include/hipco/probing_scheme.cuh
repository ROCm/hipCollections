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

#include <hipco/detail/probing_scheme_base.cuh>

#include <hip_extensions/hip_cooperative_groups_ext/amd_cooperative_groups_ext.cuh>
#include <hip/hip_cooperative_groups.h>

namespace hipco {
namespace experimental {
// todo(HIP): change back to default HIP cooperative groups if it is available
namespace cooperative_groups = hip_extensions::hip_cooperative_groups_ext;
/**
 * @brief Public linear probing scheme class.
 *
 * @note Linear probing is efficient when few collisions are present, e.g., low occupancy or low
 * multiplicity.
 *
 * @note `Hash` should be callable object type.
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 * @tparam Hash Unary callable type
 */
template <int32_t CGSize, typename Hash>
class linear_probing : private detail::probing_scheme_base<CGSize> {
 public:
  using probing_scheme_base_type =
    detail::probing_scheme_base<CGSize>;  ///< The base probe scheme type
  using probing_scheme_base_type::cg_size;

  /**
   *@brief Constructs linear probing scheme with the hasher callable.
   *
   * @param hash Hasher
   */
  __host__ __device__ constexpr linear_probing(Hash const& hash = {});

  /**
   * @brief Operator to return a probing iterator
   *
   * @tparam ProbeKey Type of probing key
   * @tparam Extent Type of extent
   *
   * @param probe_key The probing key
   * @param upper_bound Upper bound of the iteration
   * @return An iterator whose value_type is convertible to slot index type
   */
  template <typename ProbeKey, typename Extent>
  __host__ __device__ constexpr auto operator()(ProbeKey const& probe_key,
                                                Extent upper_bound) const noexcept;

  /**
   * @brief Operator to return a CG-based probing iterator
   *
   * @tparam ProbeKey Type of probing key
   * @tparam Extent Type of extent
   *
   * @param g the Cooperative Group to generate probing iterator
   * @param probe_key The probing key
   * @param upper_bound Upper bound of the iteration
   * @return An iterator whose value_type is convertible to slot index type
   */
  template <typename ProbeKey, typename Extent>
  __host__ __device__ constexpr auto operator()(
    cooperative_groups::thread_block_tile<cg_size> const& g,
    ProbeKey const& probe_key,
    Extent upper_bound) const noexcept;

 private:
  Hash hash_;
};

/**
 * @brief Public double hashing scheme class.
 *
 * @note Default probing scheme for hipco data structures. It shows superior performance over linear
 * probing especially when dealing with high multiplicty and/or high occupancy use cases.
 *
 * @note `Hash1` and `Hash2` should be callable object type.
 *
 * @note `Hash2` needs to be able to construct from an integer value to avoid secondary clustering.
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 * @tparam Hash1 Unary callable type
 * @tparam Hash2 Unary callable type
 */
template <int32_t CGSize, typename Hash1, typename Hash2 = Hash1>
class double_hashing : private detail::probing_scheme_base<CGSize> {
 public:
  using probing_scheme_base_type =
    detail::probing_scheme_base<CGSize>;  ///< The base probe scheme type
  using probing_scheme_base_type::cg_size;

  /**
   *@brief Constructs double hashing probing scheme with the two hasher callables.
   *
   * @param hash1 First hasher
   * @param hash2 Second hasher
   */
  __host__ __device__ constexpr double_hashing(Hash1 const& hash1 = {}, Hash2 const& hash2 = {1});

  /**
   * @brief Operator to return a probing iterator
   *
   * @tparam ProbeKey Type of probing key
   * @tparam Extent Type of extent
   *
   * @param probe_key The probing key
   * @param upper_bound Upper bound of the iteration
   * @return An iterator whose value_type is convertible to slot index type
   */
  template <typename ProbeKey, typename Extent>
  __host__ __device__ constexpr auto operator()(ProbeKey const& probe_key,
                                                Extent upper_bound) const noexcept;

  /**
   * @brief Operator to return a CG-based probing iterator
   *
   * @tparam ProbeKey Type of probing key
   * @tparam Extent Type of extent
   *
   * @param g the Cooperative Group to generate probing iterator
   * @param probe_key The probing key
   * @param upper_bound Upper bound of the iteration
   * @return An iterator whose value_type is convertible to slot index type
   */
  template <typename ProbeKey, typename Extent>
  __host__ __device__ constexpr auto operator()(
    cooperative_groups::thread_block_tile<cg_size> const& g,
    ProbeKey const& probe_key,
    Extent upper_bound) const noexcept;

 private:
  Hash1 hash1_;
  Hash2 hash2_;
};

}  // namespace experimental
}  // namespace hipco

#include <hipco/detail/probing_scheme_impl.inl>
