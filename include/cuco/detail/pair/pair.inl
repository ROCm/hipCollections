/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

// Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

#include <cuda/std/type_traits>
#include <cuda/std/utility>

// FIXME/WAR(HIP/AMD): We need to include libhipcxx __config here to make
// the namespace macros from libhipcxx available so that symbols defined
// inside tuple_helpers.inl land in the correct std-equivalent namespace.
#include <cuda/std/__internal/namespaces.h>

namespace cuco {

template <typename First, typename Second>
__host__ __device__ constexpr pair<First, Second>::pair(First const& f, Second const& s)
  : first{f}, second{s}
{
}

template <typename First, typename Second>
template <typename F, typename S>
__host__ __device__ constexpr pair<First, Second>::pair(pair<F, S> const& p)
  : first{p.first}, second{p.second}
{
}

template <typename F, typename S>
__host__ __device__ constexpr pair<cuda::std::decay_t<F>, cuda::std::decay_t<S>> make_pair(
  F&& f, S&& s) noexcept
{
  return pair<cuda::std::decay_t<F>, cuda::std::decay_t<S>>(cuda::std::forward<F>(f),
                                                            cuda::std::forward<S>(s));
}

template <class T1, class T2, class U1, class U2>
__host__ __device__ constexpr bool operator==(cuco::pair<T1, T2> const& lhs,
                                              cuco::pair<U1, U2> const& rhs) noexcept
{
  return lhs.first == rhs.first and lhs.second == rhs.second;
}

// NOTE(HIP/AMD): Provide `get<I>(cuco::pair<T1, T2>)` in `cuco::` so that
// C++17 structured-binding lookup finds it via ADL. The tuple_size and
// tuple_element specializations live in the std-equivalent namespace below.
template <std::size_t Index, typename T1, typename T2>
__host__ __device__ constexpr auto& get(pair<T1, T2>& p) noexcept
{
  static_assert(Index < 2);
  if constexpr (Index == 0) { return p.first; } else { return p.second; }
}
template <std::size_t Index, typename T1, typename T2>
__host__ __device__ constexpr auto const& get(pair<T1, T2> const& p) noexcept
{
  static_assert(Index < 2);
  if constexpr (Index == 0) { return p.first; } else { return p.second; }
}
template <std::size_t Index, typename T1, typename T2>
__host__ __device__ constexpr auto&& get(pair<T1, T2>&& p) noexcept
{
  static_assert(Index < 2);
  if constexpr (Index == 0) { return cuda::std::move(p.first); }
  else { return cuda::std::move(p.second); }
}

}  // namespace cuco

// NOTE(HIP/AMD): Use the libhipcxx namespace macros so the tuple_helpers.inl
// specializations land in the same std-equivalent namespace as
// cuda::std::{tuple,tuple_size,tuple_element}. The macro name changed from
// _LIBCUDACXX_BEGIN_NAMESPACE_STD_NOVERSION (libhipcxx <=2.7) to
// _CCCL_BEGIN_NAMESPACE_STD (libhipcxx 3.x).
_CCCL_BEGIN_NAMESPACE_STD
#include <cuco/detail/pair/tuple_helpers.inl>
_CCCL_END_NAMESPACE_STD
