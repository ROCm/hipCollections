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

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <vector>

namespace hipco::benchmark::defaults {

using KEY_TYPE_RANGE   = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using VALUE_TYPE_RANGE = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using INPUT_TYPE_RANGE =
  nvbench::enum_type_list<100000000, 10000000, 8000000, 5242880, 5000000, 1000000, 500000>;
using CG_SIZE_RANGE    = nvbench::enum_type_list<1, 2, 4, 8, 16, 32, 64>;
using TILE_SIZE_RANGE    = nvbench::enum_type_list<1, 2, 4, 8, 16, 32, 64>;
using BLOCK_SIZE_RANGE =  nvbench::enum_type_list<1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024>;

auto constexpr N             = 100'000'000;
auto constexpr OCCUPANCY     = 0.5;
auto constexpr MULTIPLICITY  = 8;
auto constexpr MATCHING_RATE = 0.5;
auto constexpr MAX_NOISE     = 3;
auto constexpr SKEW          = 0.5;
auto constexpr BATCH_SIZE    = 1'000'000;
auto constexpr INITIAL_SIZE  = 50'000'000;

auto const N_RANGE             = nvbench::range(10'000'000, 100'000'000, 20'000'000);
auto const N_RANGE_CACHE       = std::vector<nvbench::int64_t>{8'000, 80'000, 800'000, 8'000'000, 80'000'000};
auto const OCCUPANCY_RANGE     = nvbench::range(0.1, 0.9, 0.1);
auto const MULTIPLICITY_RANGE  = std::vector<nvbench::int64_t>{1, 2, 4, 8, 16};
auto const MATCHING_RATE_RANGE = nvbench::range(0.1, 1., 0.1);
auto const SKEW_RANGE          = nvbench::range(0.1, 1., 0.1);

}  // namespace hipco::benchmark::defaults
