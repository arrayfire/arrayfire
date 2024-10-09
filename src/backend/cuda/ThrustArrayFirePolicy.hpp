/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <backend.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <thrust/memory.h>
#include <thrust/system/cuda/execution_policy.h>

namespace arrayfire {
namespace cuda {
struct ThrustArrayFirePolicy
    : thrust::cuda::execution_policy<ThrustArrayFirePolicy> {};

template<typename T>
thrust::pair<thrust::pointer<T, ThrustArrayFirePolicy>, std::ptrdiff_t>
get_temporary_buffer(ThrustArrayFirePolicy, std::ptrdiff_t n) {
    thrust::pointer<T, ThrustArrayFirePolicy> result(
        arrayfire::cuda::memAlloc<T>(n / sizeof(T)).release());

    return thrust::make_pair(result, n);
}

template<typename Pointer>
inline void return_temporary_buffer(ThrustArrayFirePolicy, Pointer p) {
    memFree(thrust::raw_pointer_cast(p));
}

}  // namespace cuda
}  // namespace arrayfire

#if defined(_WIN32)
THRUST_NAMESPACE_BEGIN
#else
namespace thrust {
#endif
namespace cuda_cub {
template<>
__DH__ inline cudaStream_t get_stream<arrayfire::cuda::ThrustArrayFirePolicy>(
    execution_policy<arrayfire::cuda::ThrustArrayFirePolicy> &) {
#if defined(__CUDA_ARCH__)
    return 0;
#else
    return arrayfire::cuda::getActiveStream();
#endif

#if defined(_WIN32)
    THRUST_NAMESPACE_END
#else
}
#endif

    __DH__
    inline cudaError_t synchronize_stream(
        const arrayfire::cuda::ThrustArrayFirePolicy &) {
#if defined(__CUDA_ARCH__)
        return cudaSuccess;
#else
    return cudaStreamSynchronize(arrayfire::cuda::getActiveStream());
#endif
    }

}  // namespace cuda_cub
}  // namespace cuda_cub
