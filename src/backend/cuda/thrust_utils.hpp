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

#define THRUST_SELECT(fn, ...) \
    fn(arrayfire::cuda::ThrustArrayFirePolicy(), __VA_ARGS__)
#define THRUST_SELECT_OUT(res, fn, ...) \
    res = fn(arrayfire::cuda::ThrustArrayFirePolicy(), __VA_ARGS__)
