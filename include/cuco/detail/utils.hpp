/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.
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
 */

#pragma once

#include <cuco/detail/error.hpp>
#include <cuco/detail/utility/cuda.hpp>

#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace cuco {
namespace detail {

template <typename, typename = void>
inline constexpr bool has_random_access_iterator_concept = false;

template <typename Iterator>
inline constexpr bool
  has_random_access_iterator_concept<Iterator,
                                     cuda::std::void_t<typename Iterator::iterator_concept>> =
    cuda::std::is_base_of_v<cuda::std::random_access_iterator_tag,
                            typename Iterator::iterator_concept>;

// True if a single iterator is random-access, by either the C++20
// `random_access_iterator` concept or an `iterator_concept` member tag. The
// concept leg is what accepts libcu++ fancy iterators such as
// `cuda::counting_iterator`, whose legacy `iterator_category` is not
// random-access. Both legs are SFINAE-friendly, so this is safe to evaluate for
// non-iterator types (e.g. `thrust::null_type` tuple padding).
template <typename Iterator>
inline constexpr bool is_random_access_iterator =
  cuda::std::random_access_iterator<Iterator> || has_random_access_iterator_concept<Iterator>;

// NOTE(HIP/AMD): rocThrust's `zip_iterator` wrapping a libcu++ fancy iterator
// (e.g. `cuda::counting_iterator`) reports `iterator_category = input` and
// exposes no `iterator_concept`, so the wrapper itself looks non-random-access
// even when every underlying iterator is. Validate such zip iterators by their
// element iterators instead (skipping the `thrust::null_type` tuple padding).
// TODO(#101): drop this zip carve-out once rocThrust propagates cccl iterator
// categories through `zip_iterator`.
template <typename Iterator>
inline constexpr bool is_zip_iterator = false;
template <typename... Iterators>
inline constexpr bool is_zip_iterator<thrust::zip_iterator<thrust::tuple<Iterators...>>> = true;

template <typename T>
inline constexpr bool zip_element_is_random_access =
  cuda::std::is_same_v<T, thrust::null_type> || is_random_access_iterator<T>;

template <typename Iterator>
struct zip_is_random_access : cuda::std::true_type {};
template <typename... Iterators>
struct zip_is_random_access<thrust::zip_iterator<thrust::tuple<Iterators...>>>
  : cuda::std::bool_constant<(zip_element_is_random_access<Iterators> && ...)> {};

template <typename Iterator>
inline constexpr bool is_random_access =
  is_zip_iterator<Iterator> ? zip_is_random_access<Iterator>::value
                            : is_random_access_iterator<Iterator>;

template <typename Iterator>
__host__ __device__ constexpr inline index_type distance(Iterator begin, Iterator end)
{
  static_assert(is_random_access<Iterator>, "Input iterator should be a random access iterator.");
  return static_cast<index_type>(cuda::std::distance(begin, end));
}

/**
 * @brief C++17 constexpr backport of `std::lower_bound`.
 *
 * @tparam ForwardIt Type of input iterator
 * @tparam T Type of `value`
 *
 * @param first Iterator defining the start of the range to examine
 * @param last Iterator defining the start of the range to examine
 * @param value Value to compare the elements to
 *
 * @return Iterator pointing to the first element in the range [first, last) that does not satisfy
 * element < value
 */
template <class ForwardIt, class T>
constexpr ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value)
{
  using diff_type = typename std::iterator_traits<ForwardIt>::difference_type;

  ForwardIt it{};
  diff_type count = std::distance(first, last);
  diff_type step{};

  while (count > 0) {
    it   = first;
    step = count / 2;
    std::advance(it, step);

    if (static_cast<T>(*it) < value) {
      first = ++it;
      count -= step + 1;
    } else
      count = step;
  }

  return first;
}

}  // namespace detail
}  // namespace cuco
