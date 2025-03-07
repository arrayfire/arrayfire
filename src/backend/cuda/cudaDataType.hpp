/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/half.hpp>
#include <library_types.h>  // cudaDataType enum
#include <types.hpp>

namespace arrayfire {
namespace cuda {

template<typename T>
inline cudaDataType_t getType();

template<>
inline cudaDataType_t getType<float>() {
    return CUDA_R_32F;
}

template<>
inline cudaDataType_t getType<cfloat>() {
    return CUDA_C_32F;
}

template<>
inline cudaDataType_t getType<double>() {
    return CUDA_R_64F;
}

template<>
inline cudaDataType_t getType<cdouble>() {
    return CUDA_C_64F;
}

template<>
inline cudaDataType_t getType<common::half>() {
    return CUDA_R_16F;
}

template<typename T>
inline cudaDataType_t getComputeType() {
    return getType<T>();
}

template<>
inline cudaDataType_t getComputeType<common::half>() {
    cudaDataType_t algo = getType<common::half>();
    // There is probbaly a bug in nvidia cuda docs and/or drivers: According to
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmEx computeType
    // could be 32F even if A/B inputs are 16F. But CudaCompute 6.1 GPUs (for
    // example GTX10X0) dont seem to be capbale to compute at f32 when the
    // inputs are f16: results are inf if trying to do so and cublasGemmEx even
    // returns OK. At the moment let's comment out : the drawback is just that
    // the speed of f16 computation on these GPUs is very slow:
    //
    // auto dev            = getDeviceProp(getActiveDeviceId());
    // if (dev.major == // 6 && dev.minor == 1) { algo = CUDA_R_32F; }

    return algo;
}

}  // namespace cuda
}  // namespace arrayfire
