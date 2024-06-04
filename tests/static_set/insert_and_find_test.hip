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

#include <utils.hpp>

#include <hipco/static_set.cuh>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <hip_extensions/hip_cooperative_groups_ext/amd_cooperative_groups_ext.cuh>
#include <catch2/catch_template_test_macros.hpp>
namespace cg = hip_extensions::hip_cooperative_groups_ext;
template <typename Set>
__inline__ void test_insert_and_find(Set& set, std::size_t num_keys)
{
  using Key                     = typename Set::key_type;
  static auto constexpr cg_size = Set::cg_size;

  auto const keys_begin = [&]() {
    if constexpr (cg_size == 1) {
      return thrust::counting_iterator<Key>(0);
    } else {
      return thrust::make_transform_iterator(thrust::counting_iterator<Key>(0),
                                             [] __device__(auto i) { return i / cg_size; });
    }
  }();
  auto const keys_end = [&]() {
    if constexpr (cg_size == 1) {
      return keys_begin + num_keys;
    } else {
      return keys_begin + num_keys * cg_size;
    }
  }();

  auto ref = set.ref(hipco::experimental::op::insert_and_find);

  REQUIRE(hipco::test::all_of(keys_begin, keys_end, [ref] __device__(Key key) mutable {
    auto [iter, inserted] = [&]() {
      if constexpr (cg_size == 1) {
        return ref.insert_and_find(key);
      } else {
        auto const tile =
          cg::tiled_partition<cg_size>(cg::this_thread_block());
        return ref.insert_and_find(tile, key);
      }
    }();
    return inserted == true;
  }));

  SECTION("Inserting elements for the second time will always fail.")
  {
    REQUIRE(hipco::test::all_of(keys_begin, keys_end, [ref] __device__(Key key) mutable {
      auto [iter, inserted] = [&]() {
        if constexpr (cg_size == 1) {
          return ref.insert_and_find(key);
        } else {
          auto const tile =
            cg::tiled_partition<cg_size>(cg::this_thread_block());
          return ref.insert_and_find(tile, key);
        }
      }();
      return inserted == false and key == *iter;
    }));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "Insert and find",
  "",
  ((typename Key, hipco::test::probe_sequence Probe, int CGSize), Key, Probe, CGSize),
  (int32_t, hipco::test::probe_sequence::double_hashing, 1),
  (int32_t, hipco::test::probe_sequence::double_hashing, 2),
  (int64_t, hipco::test::probe_sequence::double_hashing, 1),
  (int64_t, hipco::test::probe_sequence::double_hashing, 2),
  (int32_t, hipco::test::probe_sequence::linear_probing, 1),
  (int32_t, hipco::test::probe_sequence::linear_probing, 2),
  (int64_t, hipco::test::probe_sequence::linear_probing, 1),
  (int64_t, hipco::test::probe_sequence::linear_probing, 2))
{
  constexpr std::size_t num_keys{400};

  using probe = std::conditional_t<
    Probe == hipco::test::probe_sequence::linear_probing,
    hipco::experimental::linear_probing<CGSize, hipco::default_hash_function<Key>>,
    hipco::experimental::double_hashing<CGSize, hipco::default_hash_function<Key>>>;

  auto set = hipco::experimental::static_set<Key,
                                            hipco::experimental::extent<std::size_t>,
                                            hip::thread_scope_device,
                                            thrust::equal_to<Key>,
                                            probe,
                                            hipco::cuda_allocator<std::byte>,
                                            hipco::experimental::storage<2>>{
    num_keys, hipco::empty_key<Key>{-1}};
  test_insert_and_find(set, num_keys);
}
