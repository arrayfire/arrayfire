/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <math.hpp>                     // CUDA specific math functions

#include <Param.hpp>                    // This header has the declaration of structures
                                        // that are passed onto kernel. Operator overloads
                                        // for creating Param objects from cuda::Array<T>
                                        // objects is automatic, no special work is needed.
                                        // Hence, the CUDA kernel wrapper function takes in
                                        // Param and CParam(constant version of Param) instead
                                        // of cuda::Array<T>

#include <dispatch.hpp>                 // common utility header for CUDA & OpenCL backends
                                        // has the divup macro

#include <err_cuda.hpp>                 // CUDA specific error check functions and macros

#include <debug_cuda.hpp>               // For Debug only related CUDA validations

namespace cuda
{

namespace kernel
{

static const unsigned TX = 16;          // Kernel Launch Config Values
static const unsigned TY = 16;          // Kernel Launch Config Values

template<typename T>
__global__
void exampleFuncKernel(Param<T> out, CParam<T> in, const af_someenum_t p)
{
    // kernel implementation goes here
}


template <typename T>                   // CUDA kernel wrapper function
void exampleFunc(Param<T> out, CParam<T> in, const af_someenum_t p)
{

    dim3 threads(TX, TY, 1);            // set your cuda launch config for blocks

    int blk_x = divup(out.dims[0], threads.x);
    int blk_y = divup(out.dims[1], threads.y);
    dim3 blocks(blk_x, blk_y);          // set your opencl launch config for grid

    // launch your kernel
    // One must use CUDA_LAUNCH macro to launch their kernels to ensure
    // that the kernel is launched on an appropriate stream
    //
    // Use CUDA_LAUNCH macro for launching kernels that don't use dynamic shared memory
    //
    // Use CUDA_LAUNCH_SMEM macro for launching kernsl that use dynamic shared memory
    //
    // CUDA_LAUNCH_SMEM takes in an additional parameter, size of shared memory, after
    // threads paramters, which are then followed by kernel parameters
    CUDA_LAUNCH((exampleFuncKernel<T>), blocks, threads, out, in, p);

    POST_LAUNCH_CHECK();                // Macro for post kernel launch checks
                                        // these checks are carried  ONLY IN DEBUG mode
}

}

}
