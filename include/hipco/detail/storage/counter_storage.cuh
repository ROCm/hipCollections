/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <hipco/cuda_stream_ref.hpp>
#include <hipco/detail/error.hpp>
#include <hipco/detail/storage/storage_base.cuh>
#include <hipco/extent.cuh>

#include <hip/atomic>

#include <memory>

namespace hipco {
namespace experimental {
namespace detail {
/**
 * @brief Device atomic counter storage class.
 *
 * @tparam SizeType Type of storage size
 * @tparam Scope The scope in which the counter will be used by individual threads
 * @tparam Allocator Type of allocator used for device storage
 */
template <typename SizeType, hip::thread_scope Scope, typename Allocator>
class counter_storage : public storage_base<hipco::experimental::extent<SizeType, 1>> {
 public:
 //Todo(HIP): commented this line
  // using storage_base<hipco::experimental::extent<SizeType, 1>>::capacity_;  ///< Storage size

  using size_type      = SizeType;                        ///< Size type
  using value_type     = hip::atomic<size_type, Scope>;  ///< Type of the counter
  using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<
    value_type>;  ///< Type of the allocator to (de)allocate counter
  using counter_deleter_type = custom_deleter<size_type, allocator_type>;  ///< Type of counter deleter

  /**
   * @brief Constructor of counter storage.
   *
   * @param allocator Allocator used for (de)allocating device storage
   */
  explicit constexpr counter_storage(Allocator const& allocator)
    : storage_base<hipco::experimental::extent<SizeType, 1>>{hipco::experimental::extent<size_type,
                                                                                       1>{}},
      allocator_{allocator},
      counter_deleter_{this->capacity(), allocator_},
      counter_{allocator_.allocate(this->capacity()), counter_deleter_}
  {
  }

  /**
   * @brief Asynchronously resets counter to zero.
   *
   * @param stream CUDA stream used to reset
   */
  void reset(cuda_stream_ref stream)
  {
    static_assert(sizeof(size_type) == sizeof(value_type));
    HIPCO_HIP_TRY(hipMemsetAsync(this->data(), 0, sizeof(value_type), stream));
  }

  /**
   * @brief Gets device atomic counter pointer.
   *
   * @return Pointer to the device atomic counter
   */
  [[nodiscard]] constexpr value_type* data() noexcept { return counter_.get(); }

  /**
   * @brief Gets device atomic counter pointer.
   *
   * @return Pointer to the device atomic counter
   */
  [[nodiscard]] constexpr value_type* data() const noexcept { return counter_.get(); }

  /**
   * @brief Atomically obtains the value of the device atomic counter and copies it to the host.
   *
   * @note This API synchronizes the given `stream`.
   *
   * @param stream CUDA stream used to copy device value to the host
   * @return Value of the atomic counter
   */
  [[nodiscard]] constexpr size_type load_to_host(cuda_stream_ref stream) const
  {
    size_type h_count;
    HIPCO_HIP_TRY(
      hipMemcpyAsync(&h_count, this->data(), sizeof(size_type), hipMemcpyDeviceToHost, stream));
    stream.synchronize();
    return h_count;
  }

 private:
  allocator_type allocator_;              ///< Allocator used to (de)allocate counter
  counter_deleter_type counter_deleter_;  ///< Custom counter deleter
  std::unique_ptr<value_type, counter_deleter_type> counter_;  ///< Pointer to counter storage
};

}  // namespace detail
}  // namespace experimental
}  // namespace hipco
