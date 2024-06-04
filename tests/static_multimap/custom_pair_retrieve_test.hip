/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <hipco/static_multimap.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <catch2/catch_template_test_macros.hpp>

#include "hip/hip_cooperative_groups.h"

// Custom pair equal
template <typename Key, typename Value>
struct pair_equal {
  __device__ bool operator()(const hipco::pair<Key, Value>& lhs,
                             const hipco::pair<Key, Value>& rhs) const
  {
    return lhs.first == rhs.first;
  }
};

template <uint32_t block_size,
          uint32_t cg_size,
          typename InputIt,
          typename OutputIt1,
          typename OutputIt2,
          typename OutputIt3,
          typename OutputIt4,
          typename ScanIt,
          typename viewT,
          typename PairEqual>
__global__ void custom_pair_retrieve_outer(InputIt first,
                                           InputIt last,
                                           OutputIt1 probe_key_begin,
                                           OutputIt2 probe_val_begin,
                                           OutputIt3 contained_key_begin,
                                           OutputIt4 contained_val_begin,
                                           ScanIt scan_begin,
                                           viewT view,
                                           PairEqual pair_equal)
{
  auto g        = hipco::test::cg::tiled_partition<cg_size>(hipco::test::cg::this_thread_block());
  auto tid      = block_size * blockIdx.x + threadIdx.x;
  auto pair_idx = tid / cg_size;

  while (first + pair_idx < last) {
    auto const offset = *(scan_begin + pair_idx);
    auto const pair   = *(first + pair_idx);
    view.pair_retrieve_outer(g,
                             pair,
                             probe_key_begin + offset,
                             probe_val_begin + offset,
                             contained_key_begin + offset,
                             contained_val_begin + offset,
                             pair_equal);
    pair_idx += (gridDim.x * block_size) / cg_size;
  }
}

TEMPLATE_TEST_CASE_SIG(
  "Tests of non-shared-memory pair_retrieve",
  "",
  ((typename Key, typename Value, hipco::test::probe_sequence Probe), Key, Value, Probe),
  (int32_t, int32_t, hipco::test::probe_sequence::linear_probing),
  (int32_t, int64_t, hipco::test::probe_sequence::linear_probing),
  (int64_t, int64_t, hipco::test::probe_sequence::linear_probing),
  (int32_t, int32_t, hipco::test::probe_sequence::double_hashing),
  (int32_t, int64_t, hipco::test::probe_sequence::double_hashing),
  (int64_t, int64_t, hipco::test::probe_sequence::double_hashing))
{
  constexpr std::size_t num_pairs{200};

  using probe = std::conditional_t<
    Probe == hipco::test::probe_sequence::linear_probing,
    hipco::linear_probing<1, hipco::murmurhash3_32<Key>>,
    hipco::double_hashing<8, hipco::murmurhash3_32<Key>, hipco::murmurhash3_32<Key>>>;

  hipco::static_multimap<Key, Value, hip::thread_scope_device, hipco::cuda_allocator<char>, probe>
    map{num_pairs * 2, hipco::empty_key<Key>{-1}, hipco::empty_value<Value>{-1}};
  // test_non_shmem_pair_retrieve(map, num_pairs);

  thrust::device_vector<hipco::pair<Key, Value>> d_pairs(num_pairs);

  // pair multiplicity = 2
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_pairs),
                    d_pairs.begin(),
                    [] __device__(auto i) {
                      return hipco::pair<Key, Value>{i / 2, i};
                    });

  auto pair_begin = d_pairs.begin();

  map.insert(pair_begin, pair_begin + num_pairs);

  // query pair matching rate = 50%
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_pairs),
                    pair_begin,
                    [] __device__(auto i) {
                      return hipco::pair<Key, Value>{i, i};
                    });

  // create an array of prefix sum
  thrust::device_vector<int> d_scan(num_pairs);
  auto count_begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator<int>(0),
    [] __device__(int i) { //Todo(HIP): fix 
      return i < (num_pairs / 2) ? 2 : 1; });
  thrust::exclusive_scan(thrust::device, count_begin, count_begin + num_pairs, d_scan.begin(), 0);

  auto constexpr gold_size  = 300;
  auto constexpr block_size = 128;
  auto constexpr cg_size    = map.cg_size();

  auto const grid_size = (cg_size * num_pairs + block_size - 1) / block_size;

  auto view = map.get_device_view();

  auto num = map.pair_count_outer(pair_begin, pair_begin + num_pairs, pair_equal<Key, Value>{});
  REQUIRE(num == gold_size);

  thrust::device_vector<Key> probe_keys(gold_size);
  thrust::device_vector<Value> probe_vals(gold_size);
  thrust::device_vector<Key> contained_keys(gold_size);
  thrust::device_vector<Value> contained_vals(gold_size);

  custom_pair_retrieve_outer<block_size, cg_size>
    <<<grid_size, block_size>>>(pair_begin,
                                pair_begin + num_pairs,
                                probe_keys.begin(),
                                probe_vals.begin(),
                                contained_keys.begin(),
                                contained_vals.begin(),
                                d_scan.begin(),
                                view,
                                pair_equal<Key, Value>{});

  // sort before compare
  thrust::sort(thrust::device, probe_keys.begin(), probe_keys.end());
  thrust::sort(thrust::device, probe_vals.begin(), probe_vals.end());
  thrust::sort(thrust::device, contained_keys.begin(), contained_keys.end());
  thrust::sort(thrust::device, contained_vals.begin(), contained_vals.end());

  // set gold references
  auto gold_probe         = thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0),
                                                    [] __device__(int i) {
                                                      if (i < num_pairs) { return i / 2; }
                                                      return i - (int(num_pairs) / 2);
                                                    });
  auto gold_contained_key = thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0),
                                                            [] __device__(int i) {
                                                              if (i < num_pairs / 2) { return -1; }
                                                              return (i - (int(num_pairs) / 2)) / 2;
                                                            });
  auto gold_contained_val = thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0),
                                                            [] __device__(int i) {
                                                              if (i < num_pairs / 2) { return -1; }
                                                              return i - (int(num_pairs) / 2);
                                                            });

  auto key_equal   = thrust::equal_to<Key>{};
  auto value_equal = thrust::equal_to<Value>{};

  REQUIRE(
    hipco::test::equal(probe_keys.begin(), probe_keys.begin() + gold_size, gold_probe, key_equal));

  REQUIRE(
    hipco::test::equal(probe_vals.begin(), probe_vals.begin() + gold_size, gold_probe, value_equal));

  REQUIRE(hipco::test::equal(
    contained_keys.begin(), contained_keys.begin() + gold_size, gold_contained_key, key_equal));

  REQUIRE(hipco::test::equal(
    contained_vals.begin(), contained_vals.begin() + gold_size, gold_contained_val, value_equal));
}