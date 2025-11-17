/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

// MIT License
//
// Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <test_utils.hpp>

#include <cuco/cuda_runtime_api.h>
#include <cuco/static_set.cuh>

#include <cuda/functional>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <catch2/catch_template_test_macros.hpp>

using size_type = std::size_t;

template <class Container>
__global__ void test_retrieve_if_kernel(
  Container container_ref,
  typename Container::key_type* keys_begin,
  std::size_t num_keys,
  typename Container::key_type* stencil_begin,
  typename Container::key_type* output_probe,
  typename Container::key_type* output_match,
  cuda::atomic<int, cuda::thread_scope_device>* atomic_counter)
{
  using key_type = typename Container::key_type;
  namespace cg   = cooperative_groups;

  auto const block = cg::this_thread_block();
  auto const pred  = [] __device__(key_type k) { return k % 2 == 0; };

  container_ref.template retrieve_if<128>(block,
                                 keys_begin,
                                 keys_begin + num_keys,
                                 stencil_begin,
                                 pred,
                                 output_probe,
                                 output_match,
                                 *atomic_counter);
}

template <class Container>
__global__ void test_retrieve_if_all_false_kernel(
  Container container_ref,
  typename Container::key_type* keys_begin,
  std::size_t num_keys,
  typename Container::key_type* stencil_begin,
  typename Container::key_type* output_probe,
  typename Container::key_type* output_match,
  cuda::atomic<int, cuda::thread_scope_device>* atomic_counter)
{
  using key_type = typename Container::key_type;
  namespace cg   = cooperative_groups;

  auto const block        = cg::this_thread_block();
  auto const always_false = [] __device__(key_type) { return false; };

  container_ref.template retrieve_if<128>(block,
                                 keys_begin,
                                 keys_begin + num_keys,
                                 stencil_begin,
                                 always_false,
                                 output_probe,
                                 output_match,
                                 *atomic_counter);
}

template <class Container>
__global__ void test_retrieve_if_all_true_kernel(
  Container container_ref,
  typename Container::key_type* keys_begin,
  std::size_t num_keys,
  typename Container::key_type* stencil_begin,
  typename Container::key_type* output_probe,
  typename Container::key_type* output_match,
  cuda::atomic<int, cuda::thread_scope_device>* atomic_counter)
{
  using key_type = typename Container::key_type;
  namespace cg   = cooperative_groups;

  auto const block       = cg::this_thread_block();
  auto const always_true = [] __device__(key_type) { return true; };

  container_ref.template retrieve_if<128>(block,
                                 keys_begin,
                                 keys_begin + num_keys,
                                 stencil_begin,
                                 always_true,
                                 output_probe,
                                 output_match,
                                 *atomic_counter);
}

// FIXME(HIP/AMD): dummy fixes ambiguous get_wrapper calls in catch2
TEMPLATE_TEST_CASE_SIG("static_set retrieve_if", "", ((typename Key, int dummy), Key, dummy), (int32_t, 1), (int64_t, 1))
{
  constexpr size_type num_keys{400};

  using container_type = cuco::static_set<Key>;

  container_type container{num_keys * 2, cuco::empty_key<Key>{-1}};

  auto keys_begin = thrust::counting_iterator<Key>(1);
  auto keys_end   = keys_begin + num_keys;

  container.insert(keys_begin, keys_end);

  SECTION("Testing retrieve_if with even predicate")
  {
    thrust::device_vector<Key> input_keys(keys_begin, keys_end);
    thrust::device_vector<Key> stencil_values(keys_begin, keys_end);
    thrust::device_vector<Key> probed_keys(num_keys);
    thrust::device_vector<Key> matched_keys(num_keys);

    cuda::atomic<int, cuda::thread_scope_device>* d_atomic_counter;
    CUCO_CUDA_TRY(
      cudaMalloc(&d_atomic_counter, sizeof(cuda::atomic<int, cuda::thread_scope_device>)));
    CUCO_CUDA_TRY(
      cudaMemset(d_atomic_counter, 0, sizeof(cuda::atomic<int, cuda::thread_scope_device>)));

    auto const container_ref = container.ref(cuco::op::retrieve);

    test_retrieve_if_kernel<<<1, 128>>>(container_ref,
                                        thrust::raw_pointer_cast(input_keys.data()),
                                        num_keys,
                                        thrust::raw_pointer_cast(stencil_values.data()),
                                        thrust::raw_pointer_cast(probed_keys.data()),
                                        thrust::raw_pointer_cast(matched_keys.data()),
                                        d_atomic_counter);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    int h_counter;
    CUCO_CUDA_TRY(cudaMemcpy(&h_counter, d_atomic_counter, sizeof(int), cudaMemcpyDeviceToHost));

    // Should retrieve even numbers only
    REQUIRE(h_counter > 0);
    REQUIRE(h_counter <= static_cast<int>(num_keys));

    CUCO_CUDA_TRY(cudaFree(d_atomic_counter));
  }

  SECTION("Testing retrieve_if with always false predicate")
  {
    thrust::device_vector<Key> input_keys(keys_begin, keys_end);
    thrust::device_vector<Key> stencil_values(keys_begin, keys_end);
    thrust::device_vector<Key> probed_keys(num_keys);
    thrust::device_vector<Key> matched_keys(num_keys);

    cuda::atomic<int, cuda::thread_scope_device>* d_atomic_counter;
    CUCO_CUDA_TRY(
      cudaMalloc(&d_atomic_counter, sizeof(cuda::atomic<int, cuda::thread_scope_device>)));
    CUCO_CUDA_TRY(
      cudaMemset(d_atomic_counter, 0, sizeof(cuda::atomic<int, cuda::thread_scope_device>)));

    auto const container_ref = container.ref(cuco::op::retrieve);

    test_retrieve_if_all_false_kernel<<<1, 128>>>(container_ref,
                                                  thrust::raw_pointer_cast(input_keys.data()),
                                                  num_keys,
                                                  thrust::raw_pointer_cast(stencil_values.data()),
                                                  thrust::raw_pointer_cast(probed_keys.data()),
                                                  thrust::raw_pointer_cast(matched_keys.data()),
                                                  d_atomic_counter);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    int h_counter;
    CUCO_CUDA_TRY(cudaMemcpy(&h_counter, d_atomic_counter, sizeof(int), cudaMemcpyDeviceToHost));

    // Should retrieve nothing
    REQUIRE(h_counter == 0);

    CUCO_CUDA_TRY(cudaFree(d_atomic_counter));
  }

  SECTION("Testing retrieve_if with always true predicate")
  {
    thrust::device_vector<Key> input_keys(keys_begin, keys_end);
    thrust::device_vector<Key> stencil_values(keys_begin, keys_end);
    thrust::device_vector<Key> probed_keys(num_keys);
    thrust::device_vector<Key> matched_keys(num_keys);

    cuda::atomic<int, cuda::thread_scope_device>* d_atomic_counter;
    CUCO_CUDA_TRY(
      cudaMalloc(&d_atomic_counter, sizeof(cuda::atomic<int, cuda::thread_scope_device>)));
    CUCO_CUDA_TRY(
      cudaMemset(d_atomic_counter, 0, sizeof(cuda::atomic<int, cuda::thread_scope_device>)));

    auto const container_ref = container.ref(cuco::op::retrieve);

    test_retrieve_if_all_true_kernel<<<1, 128>>>(container_ref,
                                                 thrust::raw_pointer_cast(input_keys.data()),
                                                 num_keys,
                                                 thrust::raw_pointer_cast(stencil_values.data()),
                                                 thrust::raw_pointer_cast(probed_keys.data()),
                                                 thrust::raw_pointer_cast(matched_keys.data()),
                                                 d_atomic_counter);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    int h_counter;
    CUCO_CUDA_TRY(cudaMemcpy(&h_counter, d_atomic_counter, sizeof(int), cudaMemcpyDeviceToHost));

    // Should retrieve all keys that exist in the container
    REQUIRE(h_counter == static_cast<int>(num_keys));

    CUCO_CUDA_TRY(cudaFree(d_atomic_counter));
  }
}
