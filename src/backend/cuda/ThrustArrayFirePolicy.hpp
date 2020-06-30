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
#include <thrust/execution_policy.h>

namespace cuda {
struct ThrustArrayFirePolicy
    : thrust::device_execution_policy<ThrustArrayFirePolicy> {};

namespace {
__DH__
inline cudaStream_t get_stream(ThrustArrayFirePolicy) {
#if defined(__CUDA_ARCH__)
    return 0;
#else
    return getActiveStream();
#endif
}

__DH__
inline cudaError_t synchronize_stream(ThrustArrayFirePolicy) {
#if defined(__CUDA_ARCH__)
    return cudaDeviceSynchronize();
#else
    return cudaStreamSynchronize(getActiveStream());
#endif
}
}  // namespace

template<typename T>
thrust::pair<thrust::pointer<T, ThrustArrayFirePolicy>, std::ptrdiff_t>
get_temporary_buffer(ThrustArrayFirePolicy, std::ptrdiff_t n) {
    thrust::pointer<T, ThrustArrayFirePolicy> result(
        cuda::memAlloc<T>(n / sizeof(T)).release());

    return thrust::make_pair(result, n);
}

template<typename Pointer>
inline void return_temporary_buffer(ThrustArrayFirePolicy, Pointer p) {
    memFree(thrust::raw_pointer_cast(p));
}

}  // namespace cuda
