/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <math.hpp>  // CUDA specific math functions

#include <Param.hpp>  // This header has the declaration of structures
                      // that are passed onto kernel. Operator overloads
                      // for creating Param objects from cuda::Array<T>
                      // objects is automatic, no special work is needed.
                      // Hence, the CUDA kernel wrapper function takes in
                      // Param and CParam(constant version of Param) instead
                      // of cuda::Array<T>

#include <common/dispatch.hpp>  // common utility header for CUDA & OpenCL backends
                                // has the divup macro

#include <err_cuda.hpp>  // CUDA specific error check functions and macros

#include <debug_cuda.hpp>  // For Debug only related CUDA validations

namespace cuda {

namespace kernel {

static const unsigned TX = 16;  // Kernel Launch Config Values
static const unsigned TY = 16;  // Kernel Launch Config Values

template <typename T>
__global__ void exampleFuncKernel(Param<T> c, CParam<T> a, CParam<T> b,
                                  const af_someenum_t p) {
    // get current thread global identifiers along required dimensions
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < a.dims[0] && j < a.dims[1]) {
        // if needed use strides array to compute linear index of arrays
        int src1Idx = i + j * a.strides[1];
        int src2Idx = i + j * b.strides[1];
        int dstIdx  = i + j * c.strides[1];

        T* dst        = c.ptr;
        const T* src1 = a.ptr;
        const T* src2 = b.ptr;

        // kernel algorithm goes here
        dst[dstIdx] = src1[src1Idx] + src2[src2Idx];
    }
}

template <typename T>  // CUDA kernel wrapper function
void exampleFunc(Param<T> c, CParam<T> a, CParam<T> b, const af_someenum_t p) {
    dim3 threads(TX, TY, 1);  // set your cuda launch config for blocks

    int blk_x = divup(c.dims[0], threads.x);
    int blk_y = divup(c.dims[1], threads.y);
    dim3 blocks(blk_x, blk_y);  // set your opencl launch config for grid

    // launch your kernel
    // One must use CUDA_LAUNCH macro to launch their kernels to ensure
    // that the kernel is launched on an appropriate stream
    //
    // Use CUDA_LAUNCH macro for launching kernels that don't use dynamic shared
    // memory
    //
    // Use CUDA_LAUNCH_SMEM macro for launching kernsl that use dynamic shared
    // memory
    //
    // CUDA_LAUNCH_SMEM takes in an additional parameter, size of shared memory,
    // after threads paramters, which are then followed by kernel parameters
    CUDA_LAUNCH((exampleFuncKernel<T>), blocks, threads, c, a, b, p);

    POST_LAUNCH_CHECK();  // Macro for post kernel launch checks
                          // these checks are carried  ONLY IN DEBUG mode
}

}  // namespace kernel

}  // namespace cuda
