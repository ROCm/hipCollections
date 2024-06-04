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

template <typename T1, typename T2>
struct tuple_size<hipco::pair<T1, T2>> : integral_constant<size_t, 2> {
};

template <typename T1, typename T2>
struct tuple_size<const hipco::pair<T1, T2>> : tuple_size<hipco::pair<T1, T2>> {
};

template <typename T1, typename T2>
struct tuple_size<volatile hipco::pair<T1, T2>> : tuple_size<hipco::pair<T1, T2>> {
};

template <typename T1, typename T2>
struct tuple_size<const volatile hipco::pair<T1, T2>> : tuple_size<hipco::pair<T1, T2>> {
};

template <std::size_t I, typename T1, typename T2>
struct tuple_element<I, hipco::pair<T1, T2>> {
  using type = void;
};

template <typename T1, typename T2>
struct tuple_element<0, hipco::pair<T1, T2>> {
  using type = T1;
};

template <typename T1, typename T2>
struct tuple_element<1, hipco::pair<T1, T2>> {
  using type = T2;
};

template <typename T1, typename T2>
struct tuple_element<0, const hipco::pair<T1, T2>> : tuple_element<0, hipco::pair<T1, T2>> {
};

template <typename T1, typename T2>
struct tuple_element<1, const hipco::pair<T1, T2>> : tuple_element<1, hipco::pair<T1, T2>> {
};

template <typename T1, typename T2>
struct tuple_element<0, volatile hipco::pair<T1, T2>> : tuple_element<0, hipco::pair<T1, T2>> {
};

template <typename T1, typename T2>
struct tuple_element<1, volatile hipco::pair<T1, T2>> : tuple_element<1, hipco::pair<T1, T2>> {
};

template <typename T1, typename T2>
struct tuple_element<0, const volatile hipco::pair<T1, T2>> : tuple_element<0, hipco::pair<T1, T2>> {
};

template <typename T1, typename T2>
struct tuple_element<1, const volatile hipco::pair<T1, T2>> : tuple_element<1, hipco::pair<T1, T2>> {
};

template <std::size_t I, typename T1, typename T2>
__host__ __device__ constexpr auto get(hipco::pair<T1, T2>& p) ->
  typename tuple_element<I, hipco::pair<T1, T2>>::type&
{
  static_assert(I < 2);
  if constexpr (I == 0) {
    return p.first;
  } else {
    return p.second;
  }
}

template <std::size_t I, typename T1, typename T2>
__host__ __device__ constexpr auto get(hipco::pair<T1, T2>&& p) ->
  typename tuple_element<I, hipco::pair<T1, T2>>::type&&
{
  static_assert(I < 2);
  if constexpr (I == 0) {
    return std::move(p.first);
  } else {
    return std::move(p.second);
  }
}

template <std::size_t I, typename T1, typename T2>
__host__ __device__ constexpr auto get(hipco::pair<T1, T2> const& p) ->
  typename tuple_element<I, hipco::pair<T1, T2>>::type const&
{
  static_assert(I < 2);
  if constexpr (I == 0) {
    return p.first;
  } else {
    return p.second;
  }
}

template <std::size_t I, typename T1, typename T2>
__host__ __device__ constexpr auto get(hipco::pair<T1, T2> const&& p) ->
  typename tuple_element<I, hipco::pair<T1, T2>>::type const&&
{
  static_assert(I < 2);
  if constexpr (I == 0) {
    return std::move(p.first);
  } else {
    return std::move(p.second);
  }
}