/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <math.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>

namespace cuda
{
    namespace kernel
    {
        // Kernel Launch Config Values
        static const unsigned TX = 16;
        static const unsigned TY = 16;

        template<typename T, bool D>
        inline __host__ __device__
        void diff_this(T* out, const T* in, const unsigned oMem, const unsigned iMem0,
                       const unsigned iMem1, const unsigned iMem2)
        {
            //iMem2 can never be 0
            if(D == 0) {                        // Diff1
                out[oMem] = in[iMem1] - in[iMem0];
            } else {                                // Diff2
                out[oMem] = in[iMem2] - in[iMem1] - in[iMem1] + in[iMem0];
            }
        }

        /////////////////////////////////////////////////////////////////////////////
        // 1st and 2nd Order Differential for 4D along all dimensions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, unsigned dim, bool isDiff2>
        __global__
        void diff_kernel(Param<T> out, CParam<T> in, const unsigned oElem,
                         const unsigned blocksPerMatX, const unsigned blocksPerMatY)
        {
            unsigned idz = blockIdx.x / blocksPerMatX;
            unsigned idw = blockIdx.y / blocksPerMatY;

            unsigned blockIdx_x = blockIdx.x - idz * blocksPerMatX;
            unsigned blockIdx_y = blockIdx.y - idw * blocksPerMatY;

            unsigned idx = threadIdx.x + blockIdx_x * blockDim.x;
            unsigned idy = threadIdx.y + blockIdx_y * blockDim.y;

            if(idx >= out.dims[0] ||
               idy >= out.dims[1] ||
               idz >= out.dims[2] ||
               idw >= out.dims[3])
                return;

            unsigned iMem0 = idw * in.strides[3] + idz * in.strides[2] + idy * in.strides[1] + idx;
            unsigned iMem1 = iMem0 + in.strides[dim];
            unsigned iMem2 = iMem1 + in.strides[dim];

            unsigned oMem = idw * out.strides[3] + idz * out.strides[2] + idy * out.strides[1] + idx;

            iMem2 *= isDiff2;

            diff_this<T, isDiff2>(out.ptr, in.ptr, oMem, iMem0, iMem1, iMem2);
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, unsigned dim, bool isDiff2>
        void diff(Param<T> out, CParam<T> in, const int indims)
        {
            dim3 threads(TX, TY, 1);

            if (dim == 0 && indims == 1) {
                threads = dim3(TX * TY, 1, 1);
            }

            int blocksPerMatX = divup(out.dims[0], TX);
            int blocksPerMatY = divup(out.dims[1], TY);
            dim3 blocks(blocksPerMatX * out.dims[2],
                        blocksPerMatY * out.dims[3],
                        1);

            const int oElem = out.dims[0] * out.dims[1] * out.dims[2] * out.dims[3];

            diff_kernel<T, dim, isDiff2> <<<blocks, threads>>>
                (out, in, oElem, blocksPerMatX, blocksPerMatY);

            POST_LAUNCH_CHECK();
        }
}
}
