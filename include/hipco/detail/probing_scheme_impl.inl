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

#include <hipco/detail/utils.cuh>

namespace hipco {
namespace experimental {
namespace detail {

/**
 * @brief Probing iterator class.
 *
 * @tparam Extent Type of Extent
 */
template <typename Extent>
class probing_iterator {
 public:
  using extent_type = Extent;                            ///< Extent type
  using size_type   = typename extent_type::value_type;  ///< Size type

  /**
   * @brief Constructs an probing iterator
   *
   * @param start Iteration starting point
   * @param step_size Double hashing step size
   * @param upper_bound Upper bound of the iteration
   */
  __host__ __device__ constexpr probing_iterator(size_type start,
                                                 size_type step_size,
                                                 extent_type upper_bound) noexcept
    : curr_index_{start}, step_size_{step_size}, upper_bound_{upper_bound}
  {
    // TODO: revise this API when introducing quadratic probing into hipco
  }

  /**
   * @brief Dereference operator
   *
   * @return Current slot index
   */
  __host__ __device__ constexpr auto operator*() const noexcept { return curr_index_; }

  /**
   * @brief Prefix increment operator
   *
   * @return Current iterator
   */
  __host__ __device__ constexpr auto operator++() noexcept
  {
    // TODO: step_size_ can be a build time constant (e.g. linear probing)
    //  Worth passing another extent type?
    curr_index_ = (curr_index_ + step_size_) % upper_bound_;
    return *this;
  }

  /**
   * @brief Postfix increment operator
   *
   * @return Old iterator before increment
   */
  __host__ __device__ constexpr auto operator++(int32_t) noexcept
  {
    auto temp = *this;
    ++(*this);
    return temp;
  }

 private:
  size_type curr_index_;
  size_type step_size_;
  extent_type upper_bound_;
};
}  // namespace detail

template <int32_t CGSize, typename Hash>
__host__ __device__ constexpr linear_probing<CGSize, Hash>::linear_probing(Hash const& hash)
  : hash_{hash}
{
}

template <int32_t CGSize, typename Hash>
template <typename ProbeKey, typename Extent>
__host__ __device__ constexpr auto linear_probing<CGSize, Hash>::operator()(
  ProbeKey const& probe_key, Extent upper_bound) const noexcept
{
  using size_type = typename Extent::value_type;
  return detail::probing_iterator<Extent>{
    hipco::detail::sanitize_hash<size_type>(hash_(probe_key)) % upper_bound,
    1,  // step size is 1
    upper_bound};
}

template <int32_t CGSize, typename Hash>
template <typename ProbeKey, typename Extent>
__host__ __device__ constexpr auto linear_probing<CGSize, Hash>::operator()(
  cooperative_groups::thread_block_tile<cg_size> const& g,
  ProbeKey const& probe_key,
  Extent upper_bound) const noexcept
{
  using size_type = typename Extent::value_type;
  return detail::probing_iterator<Extent>{
    hipco::detail::sanitize_hash<size_type>(hash_(probe_key) + g.thread_rank()) % upper_bound,
    cg_size,
    upper_bound};
}

template <int32_t CGSize, typename Hash1, typename Hash2>
__host__ __device__ constexpr double_hashing<CGSize, Hash1, Hash2>::double_hashing(
  Hash1 const& hash1, Hash2 const& hash2)
  : hash1_{hash1}, hash2_{hash2}
{
}

template <int32_t CGSize, typename Hash1, typename Hash2>
template <typename ProbeKey, typename Extent>
__host__ __device__ constexpr auto double_hashing<CGSize, Hash1, Hash2>::operator()(
  ProbeKey const& probe_key, Extent upper_bound) const noexcept
{
  using size_type = typename Extent::value_type;
    return detail::probing_iterator<Extent>{
    hipco::detail::sanitize_hash<size_type>(hash1_(probe_key)) % upper_bound,
    static_cast<size_type>(max(size_type{1}, //Todo(HIP): Added casting
        hipco::detail::sanitize_hash<size_type>(hash2_(probe_key)) %
          upper_bound)),  // step size in range [1, prime - 1]
    upper_bound};
}

template <int32_t CGSize, typename Hash1, typename Hash2>
template <typename ProbeKey, typename Extent>
__host__ __device__ constexpr auto double_hashing<CGSize, Hash1, Hash2>::operator()(
  cooperative_groups::thread_block_tile<cg_size> const& g,
  ProbeKey const& probe_key,
  Extent upper_bound) const noexcept
{
  using size_type = typename Extent::value_type;
  return detail::probing_iterator<Extent>{
    hipco::detail::sanitize_hash<size_type>(hash1_(probe_key) + g.thread_rank()) % upper_bound,
    static_cast<size_type>((hipco::detail::sanitize_hash<size_type>(hash2_(probe_key)) %
                              (upper_bound.value() / cg_size - 1) +
                            1) *
                           cg_size),
    upper_bound};  // TODO use fast_int operator
}
}  // namespace experimental
}  // namespace hipco
