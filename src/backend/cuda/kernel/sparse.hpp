/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <math.hpp>
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>

namespace cuda
{
    namespace kernel
    {
        static const int reps = 4;

        /////////////////////////////////////////////////////////////////////////////
        // Kernel to convert COO into Dense
        ///////////////////////////////////////////////////////////////////////////
        template<typename T>
        __global__
        void coo2dense_kernel(Param<T> output, CParam<T> values,
                              CParam<int> rowIdx, CParam<int> colIdx)
        {
            int id = blockIdx.x * blockDim.x * reps + threadIdx.x;
            if(id >= values.dims[0])
                return;

            for(int i = threadIdx.x; i <= reps * blockDim.x; i += blockDim.x) {
                if(i >= values.dims[0])
                    return;

                T   v = values.ptr[i];
                int r = rowIdx.ptr[i];
                int c = colIdx.ptr[i];

                int offset = r + c * output.strides[1];

                output.ptr[offset] = v;
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T>
        void coo2dense(Param<T> output, CParam<T> values, CParam<int> rowIdx, CParam<int> colIdx)
        {
            dim3 threads(256, 1, 1);

            dim3 blocks(divup(output.dims[0], threads.x * reps), 1, 1);

            CUDA_LAUNCH((coo2dense_kernel<T>), blocks, threads, output, values, rowIdx, colIdx);

            POST_LAUNCH_CHECK();
        }
    }
}
