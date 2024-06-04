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

#include <defaults.hpp>
#include <utils.hpp>

#include <hipco/static_map.cuh>
#include <hipco/utility/key_generator.hpp>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/transform.h>

using namespace hipco::benchmark;
using namespace hipco::utility;

/**
 * @brief A benchmark evaluating `hipco::static_map::insert` performance
 */
template <typename Key,
          typename Value,
          typename Dist,
          nvbench::int32_t BlockSize>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> static_map_insert(
  nvbench::state& state,
  nvbench::type_list<Key,
                     Value,
                     Dist,
                     nvbench::enum_type<BlockSize>>)
{
  using pair_type = hipco::pair<Key, Value>;

  auto const num_keys  = state.get_int64_or_default("NumInputs", defaults::N);
  auto const occupancy = state.get_float64_or_default("Occupancy", defaults::OCCUPANCY);

  std::size_t const size = num_keys / occupancy;

  thrust::device_vector<Key> keys(num_keys);

  key_generator gen;
  gen.generate(dist_from_state<Dist>(state), keys.begin(), keys.end());

  thrust::device_vector<pair_type> pairs(num_keys);
  thrust::transform(keys.begin(), keys.end(), pairs.begin(), [] __device__(Key const& key) {
    return pair_type(key, {});
  });

  state.add_element_count(num_keys);

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      hipco::static_map<Key,
                       Value,
                       hip::thread_scope_device,
                       hipco::cuda_allocator<char>,
                       HIPCO_TILE_SIZE,
                       BlockSize>
        map{size, hipco::empty_key<Key>{-1}, hipco::empty_value<Value>{-1}, {}, launch.get_stream()};

      timer.start();
      map.insert(pairs.begin(), pairs.end(), {}, {}, launch.get_stream());
      timer.stop();
    });
}

template <typename Key,
          typename Value,
          typename Dist,
          nvbench::int32_t BlockSize>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> static_map_insert(
  nvbench::state& state,
  nvbench::type_list<Key,
                     Value,
                     Dist,
                     nvbench::enum_type<BlockSize>>)
{
  state.skip("Key should be the same type as Value.");
}

NVBENCH_BENCH_TYPES(static_map_insert,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      defaults::VALUE_TYPE_RANGE,
                                      nvbench::type_list<distribution::uniform>,
                                      defaults::BLOCK_SIZE_RANGE))
  .set_name("static_map_insert_uniform_multiplicity")
  .set_type_axes_names({
    "Key", "Value", "Distribution", "BlockSize"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("Multiplicity", defaults::MULTIPLICITY_RANGE);

NVBENCH_BENCH_TYPES(static_map_insert,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      defaults::VALUE_TYPE_RANGE,
                                      nvbench::type_list<distribution::unique>,
                                      defaults::BLOCK_SIZE_RANGE))
  .set_name("static_map_insert_unique_occupancy")
  .set_type_axes_names({
    "Key", "Value", "Distribution", "BlockSize"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_float64_axis("Occupancy", defaults::OCCUPANCY_RANGE);

NVBENCH_BENCH_TYPES(static_map_insert,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      defaults::VALUE_TYPE_RANGE,
                                      nvbench::type_list<distribution::gaussian>,
                                      defaults::BLOCK_SIZE_RANGE))
  .set_name("static_map_insert_gaussian_skew")
  .set_type_axes_names({
    "Key", "Value", "Distribution", "BlockSize"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_float64_axis("Skew", defaults::SKEW_RANGE);
