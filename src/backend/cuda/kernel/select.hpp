/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/dispatch.hpp>
#include <Param.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <math.hpp>

namespace cuda
{
    namespace kernel
    {

        static const uint DIMX = 32;
        static const uint DIMY =  8;
        static const int REPEAT = 64;

        __device__ __host__
        int getOffset(dim_t *dims, dim_t *strides, dim_t *refdims, int ids[4])
        {
            int off = 0;
            off += ids[3] * (dims[3] == refdims[3]) * strides[3];
            off += ids[2] * (dims[2] == refdims[2]) * strides[2];
            off += ids[1] * (dims[1] == refdims[1]) * strides[1];
            return off;
        }

        template<typename T, bool is_same>
        __global__
        void select_kernel(Param<T> out, CParam<char> cond,
                           CParam<T> a, CParam<T> b, int blk_x, int blk_y)
        {
            const int idz = blockIdx.x / blk_x;
            const int idw = (blockIdx.y + blockIdx.z * gridDim.y) / blk_y;


            const int blockIdx_x = blockIdx.x - idz * blk_x;
            const int blockIdx_y = (blockIdx.y + blockIdx.z * gridDim.y) - idw * blk_y;

            const int idy = blockIdx_y * blockDim.y + threadIdx.y;
            const int idx0 = blockIdx_x * blockDim.x + threadIdx.x;

            if (idw >= out.dims[3] ||
                idz >= out.dims[2] ||
                idy >= out.dims[1])  {
                return;
            }

            const int off = idw * out.strides[3] + idz * out.strides[2] + idy * out.strides[1];
            T *optr = out.ptr + off;

            const T *aptr = a.ptr;
            const T *bptr = b.ptr;
            const char *cptr = cond.ptr;

            int ids[] = {idx0, idy, idz, idw};
            aptr += getOffset(a.dims, a.strides, out.dims, ids);
            bptr += getOffset(b.dims, b.strides, out.dims, ids);
            cptr += getOffset(cond.dims, cond.strides, out.dims, ids);

            if (is_same) {
                for (int idx = idx0; idx < out.dims[0]; idx += blockDim.x * blk_x) {
                    optr[idx] = cptr[idx] ? aptr[idx] : bptr[idx];
                }
            } else {
                bool csame = cond.dims[0] == out.dims[0];
                bool asame = a.dims[0] == out.dims[0];
                bool bsame = b.dims[0] == out.dims[0];
                for (int idx = idx0; idx < out.dims[0]; idx += blockDim.x * blk_x) {
                    optr[idx] = cptr[csame * idx] ? aptr[asame * idx] : bptr[bsame * idx];
                }
            }
        }

        template<typename T>
        void select(Param<T> out, CParam<char> cond, CParam<T> a, CParam<T> b, int ndims)
        {
            bool is_same = true;
            for (int i = 0; i < 4; i++) {
                is_same &= (a.dims[i] == b.dims[i]);
            }

            dim3 threads(DIMX, DIMY);

            if (ndims == 1) {
                threads.x *= threads.y;
                threads.y = 1;
            }

            int blk_x = divup(out.dims[0], REPEAT * threads.x);
            int blk_y = divup(out.dims[1], threads.y);


            dim3 blocks(blk_x * out.dims[2],
                        blk_y * out.dims[3]);

            const int maxBlocksY = cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
            blocks.z = divup(blocks.y, maxBlocksY);
            blocks.y = divup(blocks.y, blocks.z);

            if (is_same) {
                CUDA_LAUNCH((select_kernel<T, true>), blocks, threads,
                            out, cond, a, b, blk_x, blk_y);
            } else {
                CUDA_LAUNCH((select_kernel<T, false>), blocks, threads,
                            out, cond, a, b, blk_x, blk_y);
            }

        }

        template<typename T, bool flip>
        __global__
        void select_scalar_kernel(Param<T> out, CParam<char> cond,
                                  CParam<T> a, T b, int blk_x, int blk_y)
        {
            const int idz = blockIdx.x / blk_x;
            const int idw = (blockIdx.y + blockIdx.z * gridDim.y) / blk_y;

            const int blockIdx_x = blockIdx.x - idz * blk_x;
            const int blockIdx_y = (blockIdx.y + blockIdx.z * gridDim.y) - idw * blk_y;

            const int idx0 = blockIdx_x * blockDim.x + threadIdx.x;
            const int idy = blockIdx_y * blockDim.y + threadIdx.y;

            const int off = idw * out.strides[3] + idz * out.strides[2] + idy * out.strides[1];

            T *optr = out.ptr + off;

            const T *aptr = a.ptr;
            const char *cptr = cond.ptr;

            int ids[] = {idx0, idy, idz, idw};
            aptr += getOffset(a.dims, a.strides, out.dims, ids);
            cptr += getOffset(cond.dims, cond.strides, out.dims, ids);

            if (idw >= out.dims[3] ||
                idz >= out.dims[2] ||
                idy >= out.dims[1]) {
                return;
            }

            for (int idx = idx0; idx < out.dims[0]; idx += blockDim.x * blk_x) {
                optr[idx] = ((cptr[idx]) ^ flip) ? aptr[idx] : b;
            }
        }

        template<typename T, bool flip>
        void select_scalar(Param<T> out, CParam<char> cond, CParam<T> a, const double b, int ndims)
        {
            dim3 threads(DIMX, DIMY);

            if (ndims == 1) {
                threads.x *= threads.y;
                threads.y = 1;
            }

            int blk_x = divup(out.dims[0], REPEAT * threads.x);
            int blk_y = divup(out.dims[1], threads.y);


            dim3 blocks(blk_x * out.dims[2],
                        blk_y * out.dims[3]);

            CUDA_LAUNCH((select_scalar_kernel<T, flip>), blocks, threads,
                        out, cond, a, scalar<T>(b), blk_x, blk_y);

        }
    }
}
