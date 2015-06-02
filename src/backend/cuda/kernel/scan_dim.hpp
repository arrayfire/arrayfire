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

    template<typename Ti, typename To, af_op_t op, int dim, bool isFinalPass, uint DIMY>
    __global__
    static void scan_dim_kernel(Param<To> out,
                                Param<To> tmp,
                                CParam<Ti>  in,
                                uint blocks_x,
                                uint blocks_y,
                                uint blocks_dim,
                                uint lim)
    {
        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;
        const int tid  = tidy * THREADS_X + tidx;

        const int zid = blockIdx.x / blocks_x;
        const int wid = blockIdx.y / blocks_y;
        const int blockIdx_x = blockIdx.x - (blocks_x) * zid;
        const int blockIdx_y = blockIdx.y - (blocks_y) * wid;
        const int xid = blockIdx_x * blockDim.x + tidx;
        const int yid = blockIdx_y; // yid  of output. updated for input later.

        int ids[4] = {xid, yid, zid, wid};

        const Ti *iptr = in.ptr;
        To *optr = out.ptr;
        To *tptr = tmp.ptr;

        // There is only one element per block for out
        // There are blockDim.y elements per block for in
        // Hence increment ids[dim] just after offseting out and before offsetting in
        tptr += ids[3] * tmp.strides[3] + ids[2] * tmp.strides[2] + ids[1] * tmp.strides[1] + ids[0];
        const int blockIdx_dim = ids[dim];

        ids[dim] = ids[dim] * blockDim.y * lim + tidy;
        optr  += ids[3] * out.strides[3] + ids[2] * out.strides[2] + ids[1] * out.strides[1] + ids[0];
        iptr  += ids[3] *  in.strides[3] + ids[2] *  in.strides[2] + ids[1] *  in.strides[1] + ids[0];
        int id_dim = ids[dim];
        const int out_dim = out.dims[dim];

        bool is_valid =
            (ids[0] < out.dims[0]) &&
            (ids[1] < out.dims[1]) &&
            (ids[2] < out.dims[2]) &&
            (ids[3] < out.dims[3]);

        const int ostride_dim = out.strides[dim];
        const int istride_dim =  in.strides[dim];

        __shared__ To s_val[THREADS_X * DIMY * 2];
        __shared__ To s_tmp[THREADS_X];
        To *sptr =  s_val + tid;

        Transform<Ti, To, op> transform;
        Binary<To, op> binop;

        const To init = binop.init();
        To val = init;

        const bool isLast = (tidy == (DIMY - 1));

        for (int k = 0; k < lim; k++) {

            if (isLast) s_tmp[tidx] = val;

            bool cond = (is_valid) && (id_dim < out_dim);
            val = cond ? transform(*iptr) : init;
            *sptr = val;
            __syncthreads();

            int start = 0;
#pragma unroll
            for (int off = 1; off < DIMY; off *= 2) {

                if (tidy >= off) val = binop(val, sptr[(start - off) * THREADS_X]);
                start = DIMY - start;
                sptr[start * THREADS_X] = val;

                __syncthreads();
            }

            val = binop(val, s_tmp[tidx]);
            __syncthreads();
            if (cond) *optr = val;

            id_dim += blockDim.y;
            iptr += blockDim.y * istride_dim;
            optr += blockDim.y * ostride_dim;
        }

        if (!isFinalPass &&
            is_valid &&
            (blockIdx_dim < tmp.dims[dim]) &&
            isLast) {
            *tptr = val;
            }
    }

    template<typename To, af_op_t op, int dim>
    __global__
    static void bcast_dim_kernel(Param<To> out,
                                 CParam<To> tmp,
                                 uint blocks_x,
                                 uint blocks_y,
                                 uint blocks_dim,
                                 uint lim)
    {
        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int zid = blockIdx.x / blocks_x;
        const int wid = blockIdx.y / blocks_y;
        const int blockIdx_x = blockIdx.x - (blocks_x) * zid;
        const int blockIdx_y = blockIdx.y - (blocks_y) * wid;
        const int xid = blockIdx_x * blockDim.x + tidx;
        const int yid = blockIdx_y; // yid  of output. updated for input later.

        int ids[4] = {xid, yid, zid, wid};

        const To *tptr = tmp.ptr;
        To *optr = out.ptr;

        // There is only one element per block for out
        // There are blockDim.y elements per block for in
        // Hence increment ids[dim] just after offseting out and before offsetting in
        tptr += ids[3] * tmp.strides[3] + ids[2] * tmp.strides[2] + ids[1] * tmp.strides[1] + ids[0];
        const int blockIdx_dim = ids[dim];

        ids[dim] = ids[dim] * blockDim.y * lim + tidy;
        optr  += ids[3] * out.strides[3] + ids[2] * out.strides[2] + ids[1] * out.strides[1] + ids[0];
        const int id_dim = ids[dim];
        const int out_dim = out.dims[dim];

        bool is_valid =
            (ids[0] < out.dims[0]) &&
            (ids[1] < out.dims[1]) &&
            (ids[2] < out.dims[2]) &&
            (ids[3] < out.dims[3]);

        if (!is_valid) return;
        if (blockIdx_dim == 0) return;

        To accum = *(tptr - tmp.strides[dim]);

        Binary<To, op> binop;
        const int ostride_dim = out.strides[dim];

        for (int k = 0, id = id_dim;
             is_valid && k < lim && (id < out_dim);
             k++, id += blockDim.y) {

            *optr = binop(*optr,accum);
            optr += blockDim.y * ostride_dim;
        }
    }

    template<typename Ti, typename To, af_op_t op, int dim, bool isFinalPass>
    static void scan_dim_launcher(Param<To> out,
                           Param<To> tmp,
                           CParam<Ti> in,
                           const uint threads_y,
                           const uint blocks_all[4])
    {
        dim3 threads(THREADS_X, threads_y);

        dim3 blocks(blocks_all[0] * blocks_all[2],
                    blocks_all[1] * blocks_all[3]);

        uint lim = divup(out.dims[dim], (threads_y * blocks_all[dim]));

        switch (threads_y) {
        case 8:
            (scan_dim_kernel<Ti, To, op, dim, isFinalPass, 8>)<<<blocks, threads>>>(
                out, tmp, in, blocks_all[0], blocks_all[1], blocks_all[dim], lim); break;
        case 4:
            (scan_dim_kernel<Ti, To, op, dim, isFinalPass, 4>)<<<blocks, threads>>>(
                out, tmp, in, blocks_all[0], blocks_all[1], blocks_all[dim], lim); break;
        case 2:
            (scan_dim_kernel<Ti, To, op, dim, isFinalPass, 2>)<<<blocks, threads>>>(
                out, tmp, in, blocks_all[0], blocks_all[1], blocks_all[dim], lim); break;
        case 1:
            (scan_dim_kernel<Ti, To, op, dim, isFinalPass, 1>)<<<blocks, threads>>>(
                out, tmp, in, blocks_all[0], blocks_all[1], blocks_all[dim], lim); break;
        }

        POST_LAUNCH_CHECK();
    }



    template<typename To, af_op_t op, int dim>
    static void bcast_dim_launcher(Param<To> out,
                                   CParam<To> tmp,
                                   const uint threads_y,
                                   const uint blocks_all[4])
    {

        dim3 threads(THREADS_X, threads_y);

        dim3 blocks(blocks_all[0] * blocks_all[2],
                    blocks_all[1] * blocks_all[3]);

        uint lim = divup(out.dims[dim], (threads_y * blocks_all[dim]));

        (bcast_dim_kernel<To, op, dim>)<<<blocks, threads>>>(
            out, tmp, blocks_all[0], blocks_all[1], blocks_all[dim], lim);

        POST_LAUNCH_CHECK();
    }

    template<typename Ti, typename To, af_op_t op, int dim>
    static void scan_dim(Param<To> out, CParam<Ti> in)
    {
        uint threads_y = std::min(THREADS_Y, nextpow2(out.dims[dim]));
        uint threads_x = THREADS_X;

        uint blocks_all[] = {divup(out.dims[0], threads_x),
                             out.dims[1], out.dims[2], out.dims[3]};

        blocks_all[dim] = divup(out.dims[dim], threads_y * REPEAT);

        if (blocks_all[dim] == 1) {

            scan_dim_launcher<Ti, To, op, dim, true>(out, out, in,
                                                     threads_y,
                                                     blocks_all);

        } else {

            Param<To> tmp = out;

            tmp.dims[dim] = blocks_all[dim];
            tmp.strides[0] = 1;
            for (int k = 1; k < 4; k++) tmp.strides[k] = tmp.strides[k - 1] * tmp.dims[k - 1];

            int tmp_elements = tmp.strides[3] * tmp.dims[3];
            tmp.ptr = memAlloc<To>(tmp_elements);

            scan_dim_launcher<Ti, To, op, dim, false>(out, tmp, in,
                                                      threads_y,
                                                      blocks_all);

            int bdim = blocks_all[dim];
            blocks_all[dim] = 1;

            //FIXME: Is there an alternative to the if condition ?
            if (op == af_notzero_t) {
                scan_dim_launcher<To, To, af_add_t, dim, true>(tmp, tmp, tmp,
                                                               threads_y,
                                                               blocks_all);
            } else {
                scan_dim_launcher<To, To,       op, dim, true>(tmp, tmp, tmp,
                                                               threads_y,
                                                               blocks_all);
            }

            blocks_all[dim] = bdim;
            bcast_dim_launcher<To, op, dim>(out, tmp, threads_y, blocks_all);

            memFree(tmp.ptr);
        }
    }

}
}
