/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <ops.hpp>
#include <backend.hpp>
#include <Param.hpp>
#include <dispatch.hpp>
#include <math.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <memory.hpp>
#include "config.hpp"

namespace cuda
{
namespace kernel
{
    template<typename Ti, typename To, af_op_t op, bool isFinalPass, uint DIMX>
    __global__
    static void scan_first_kernel(Param<To> out,
                                  Param<To> tmp,
                                  CParam<Ti>  in,
                                  uint blocks_x,
                                  uint blocks_y,
                                  uint lim)
    {
        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int zid = blockIdx.x / blocks_x;
        const int wid = blockIdx.y / blocks_y;
        const int blockIdx_x = blockIdx.x - (blocks_x) * zid;
        const int blockIdx_y = blockIdx.y - (blocks_y) * wid;
        const int xid = blockIdx_x * blockDim.x * lim + tidx;
        const int yid = blockIdx_y * blockDim.y + tidy;

        bool cond_yzw = (yid < out.dims[1]) && (zid < out.dims[2]) && (wid < out.dims[3]);

        if (!cond_yzw) return; // retire warps early

        const Ti *iptr = in.ptr;
        To *optr = out.ptr;
        To *tptr = tmp.ptr;

        iptr += wid *  in.strides[3] + zid *  in.strides[2] + yid *  in.strides[1];
        optr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];
        tptr += wid * tmp.strides[3] + zid * tmp.strides[2] + yid * tmp.strides[1];


        const int DIMY = THREADS_PER_BLOCK / DIMX;
        const int SHARED_MEM_SIZE = (2 * DIMX + 1) * (DIMY);

        __shared__ To s_val[SHARED_MEM_SIZE];
        __shared__ To s_tmp[DIMY];

        To *sptr = s_val + tidy * (2 * DIMX + 1);

        Transform<Ti, To, op> transform;
        Binary<To, op> binop;

        const To init = binop.init();
        int id = xid;
        To val = init;

        const bool isLast = (tidx == (DIMX - 1));

        for (int k = 0; k < lim; k++) {

            if (isLast) s_tmp[tidy] = val;

            bool cond = ((id < out.dims[0]));
            val = cond ? transform(iptr[id]) : init;
            sptr[tidx] = val;
            __syncthreads();


            int start = 0;
#pragma unroll
            for (int off = 1; off < DIMX; off *= 2) {

                if (tidx >= off) val = binop(val, sptr[(start - off) + tidx]);
                start = DIMX - start;
                sptr[start + tidx] = val;

                __syncthreads();
            }

            val = binop(val, s_tmp[tidy]);
            if (cond) optr[id] = val;
            id += blockDim.x;
            __syncthreads();
        }

        if (!isFinalPass && isLast) {
            tptr[blockIdx_x] = val;
        }
    }

    template<typename To, af_op_t op>
    __global__
    static void bcast_first_kernel(Param<To> out,
                                   CParam<To> tmp,
                                   uint blocks_x,
                                   uint blocks_y,
                                   uint lim)
    {
        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int zid = blockIdx.x / blocks_x;
        const int wid = blockIdx.y / blocks_y;
        const int blockIdx_x = blockIdx.x - (blocks_x) * zid;
        const int blockIdx_y = blockIdx.y - (blocks_y) * wid;
        const int xid = blockIdx_x * blockDim.x * lim + tidx;
        const int yid = blockIdx_y * blockDim.y + tidy;

        if (blockIdx_x == 0) return;

        bool cond = (yid < out.dims[1]) && (zid < out.dims[2]) && (wid < out.dims[3]);
        if (!cond) return;

        To *optr = out.ptr;
        const To *tptr = tmp.ptr;

        optr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];
        tptr += wid * tmp.strides[3] + zid * tmp.strides[2] + yid * tmp.strides[1];

        Binary<To, op> binop;
        To accum = tptr[blockIdx_x - 1];

        for (int k = 0, id = xid;
             k < lim && id < out.dims[0];
             k++, id += blockDim.x) {

            optr[id] = binop(accum, optr[id]);
        }

    }

    template<typename Ti, typename To, af_op_t op, bool isFinalPass>
    static void scan_first_launcher(Param<To> out,
                             Param<To> tmp,
                             CParam<Ti> in,
                             const uint blocks_x,
                             const uint blocks_y,
                             const uint threads_x)
    {

        dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
        dim3 blocks(blocks_x * out.dims[2],
                    blocks_y * out.dims[3]);

        uint lim = divup(out.dims[0], (threads_x * blocks_x));

        switch (threads_x) {
        case 32:
            (scan_first_kernel<Ti, To, op, isFinalPass,  32>)<<<blocks, threads>>>(
                out, tmp, in, blocks_x, blocks_y, lim); break;
        case 64:
            (scan_first_kernel<Ti, To, op, isFinalPass,  64>)<<<blocks, threads>>>(
                out, tmp, in, blocks_x, blocks_y, lim); break;
        case 128:
            (scan_first_kernel<Ti, To, op, isFinalPass,  128>)<<<blocks, threads>>>(
                out, tmp, in, blocks_x, blocks_y, lim); break;
        case 256:
            (scan_first_kernel<Ti, To, op, isFinalPass,  256>)<<<blocks, threads>>>(
                out, tmp, in, blocks_x, blocks_y, lim); break;
        }

        POST_LAUNCH_CHECK();
    }



    template<typename To, af_op_t op>
    static void bcast_first_launcher(Param<To> out,
                                     CParam<To> tmp,
                                     const uint blocks_x,
                                     const uint blocks_y,
                                     const uint threads_x)
    {

        dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
        dim3 blocks(blocks_x * out.dims[2],
                    blocks_y * out.dims[3]);

        uint lim = divup(out.dims[0], (threads_x * blocks_x));

        (bcast_first_kernel<To, op>)<<<blocks, threads>>>(
            out, tmp, blocks_x, blocks_y, lim);

        POST_LAUNCH_CHECK();
    }

    template<typename Ti, typename To, af_op_t op>
    static void scan_first(Param<To> out, CParam<Ti> in)
    {
        uint threads_x = nextpow2(std::max(32u, (uint)out.dims[0]));
        threads_x = std::min(threads_x, THREADS_PER_BLOCK);
        uint threads_y = THREADS_PER_BLOCK / threads_x;

        uint blocks_x = divup(out.dims[0], threads_x * REPEAT);
        uint blocks_y = divup(out.dims[1], threads_y);

        if (blocks_x == 1) {

            scan_first_launcher<Ti, To, op, true>(out, out, in,
                                                  blocks_x, blocks_y,
                                                  threads_x);

        } else {

            Param<To> tmp = out;

            tmp.dims[0] = blocks_x;
            tmp.strides[0] = 1;
            for (int k = 1; k < 4; k++) tmp.strides[k] = tmp.strides[k - 1] * tmp.dims[k - 1];

            int tmp_elements = tmp.strides[3] * tmp.dims[3];
            tmp.ptr = memAlloc<To>(tmp_elements);

            scan_first_launcher<Ti, To, op, false>(out, tmp, in,
                                                   blocks_x, blocks_y,
                                                   threads_x);

            //FIXME: Is there an alternative to the if condition ?
            if (op == af_notzero_t) {
                scan_first_launcher<To, To, af_add_t, true>(tmp, tmp, tmp,
                                                             1, blocks_y,
                                                             threads_x);
            } else {
                scan_first_launcher<To, To,       op, true>(tmp, tmp, tmp,
                                                            1, blocks_y,
                                                            threads_x);
            }

            bcast_first_launcher<To, op>(out, tmp, blocks_x, blocks_y, threads_x);

            memFree(tmp.ptr);
        }
    }

}
}
