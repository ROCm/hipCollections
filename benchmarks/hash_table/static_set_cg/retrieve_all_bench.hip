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

#include <defaults.hpp>
#include <utils.hpp>

#include <hipco/static_set.cuh>
#include <hipco/utility/key_generator.hpp>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>

using namespace hipco::benchmark;
using namespace hipco::utility;

/**
 * @brief A benchmark evaluating `hipco::static_set::retrieve_all` performance
 */
template <typename Key, typename Dist, nvbench::int32_t CGSize>
void static_set_retrieve_all(nvbench::state& state,
                             nvbench::type_list<Key, Dist, nvbench::enum_type<CGSize>>)
{
  auto const num_keys  = state.get_int64_or_default("NumInputs", defaults::N);
  auto const occupancy = state.get_float64_or_default("Occupancy", defaults::OCCUPANCY);

  std::size_t const size = num_keys / occupancy;

  thrust::device_vector<Key> keys(num_keys);

  key_generator gen;
  gen.generate(dist_from_state<Dist>(state), keys.begin(), keys.end());

  hipco::experimental::static_set<Key,
                                 hipco::experimental::extent<std::size_t>,
                                 hip::thread_scope_device,
                                 thrust::equal_to<Key>,
                                 hipco::experimental::double_hashing<CGSize,  // CG size
                                                                    hipco::murmurhash3_32<Key>,
                                                                    hipco::murmurhash3_32<Key>>,
                                 hipco::cuda_allocator<std::byte>,
                                 hipco::experimental::storage<1>>
    set{size, hipco::empty_key<Key>{-1}};
  set.insert(keys.begin(), keys.end());

  thrust::device_vector<Key> result(num_keys);

  state.add_element_count(num_keys);
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto end = set.retrieve_all(result.begin(), {launch.get_stream()});
  });
}

NVBENCH_BENCH_TYPES(static_set_retrieve_all,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      nvbench::type_list<distribution::unique>,
                                      defaults::CG_SIZE_RANGE))
  .set_name("static_set_retrieve_all_unique_occupancy")
  .set_type_axes_names({"Key", "Distribution", "ProbingCGSize"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_float64_axis("Occupancy", defaults::OCCUPANCY_RANGE);
