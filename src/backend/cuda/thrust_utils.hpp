/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <ThrustArrayFirePolicy.hpp>
#include <thrust/system/cuda/detail/par.h>
#include <thrust/version.h>
#include <ThrustAllocator.cuh>

namespace arrayfire {
namespace cuda {
template<typename T>
using ThrustVector = thrust::device_vector<T, ThrustAllocator<T>>;
}  // namespace cuda
}  // namespace arrayfire

#if THRUST_MAJOR_VERSION >= 1 && THRUST_MINOR_VERSION >= 8

#define THRUST_SELECT(fn, ...) \
    fn(arrayfire::cuda::ThrustArrayFirePolicy(), __VA_ARGS__)
#define THRUST_SELECT_OUT(res, fn, ...) \
    res = fn(arrayfire::cuda::ThrustArrayFirePolicy(), __VA_ARGS__)

#else

#define THRUST_SELECT(fn, ...)                                                 \
    do {                                                                       \
        CUDA_CHECK(cudaStreamSynchronize(arrayfire::cuda::getActiveStream())); \
        fn(__VA_ARGS__);                                                       \
    } while (0)

#define THRUST_SELECT_OUT(res, fn, ...)                                        \
    do {                                                                       \
        CUDA_CHECK(cudaStreamSynchronize(arrayfire::cuda::getActiveStream())); \
        res = fn(__VA_ARGS__);                                                 \
    } while (0)

#endif
