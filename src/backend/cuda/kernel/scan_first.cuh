/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <backend.hpp>
#include <common/Binary.hpp>
#include <common/Transform.hpp>
#include <math.hpp>

namespace arrayfire {
namespace cuda {

template<typename Ti, typename To, af_op_t op, bool isFinalPass, uint DIMX,
         bool inclusive_scan>
__global__ void scan_first(Param<To> out, Param<To> tmp, CParam<Ti> in,
                           uint blocks_x, uint blocks_y, uint lim) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int zid        = blockIdx.x / blocks_x;
    const int wid        = (blockIdx.y + blockIdx.z * gridDim.y) / blocks_y;
    const int blockIdx_x = blockIdx.x - (blocks_x)*zid;
    const int blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - (blocks_y)*wid;
    const int xid = blockIdx_x * blockDim.x * lim + tidx;
    const int yid = blockIdx_y * blockDim.y + tidy;

    bool cond_yzw =
        (yid < out.dims[1]) && (zid < out.dims[2]) && (wid < out.dims[3]);

    if (!cond_yzw) return;  // retire warps early

    const Ti *iptr = in.ptr;
    To *optr       = out.ptr;
    To *tptr       = tmp.ptr;

    iptr += wid * in.strides[3] + zid * in.strides[2] + yid * in.strides[1];
    optr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];
    tptr += wid * tmp.strides[3] + zid * tmp.strides[2] + yid * tmp.strides[1];

    const int DIMY            = THREADS_PER_BLOCK / DIMX;
    const int SHARED_MEM_SIZE = (2 * DIMX + 1) * (DIMY);

    __shared__ To s_val[SHARED_MEM_SIZE];
    __shared__ To s_tmp[DIMY];

    To *sptr = s_val + tidy * (2 * DIMX + 1);

    common::Transform<Ti, To, op> transform;
    common::Binary<To, op> binop;

    const To init = common::Binary<To, op>::init();
    int id        = xid;
    To val        = init;

    const bool isLast = (tidx == (DIMX - 1));

    for (int k = 0; k < lim; k++) {
        if (isLast) s_tmp[tidy] = val;

        bool cond  = (id < out.dims[0]);
        val        = cond ? transform(iptr[id]) : init;
        sptr[tidx] = val;
        __syncthreads();

        int start = 0;
#pragma unroll
        for (int off = 1; off < DIMX; off *= 2) {
            if (tidx >= off) val = binop(val, sptr[(start - off) + tidx]);
            start              = DIMX - start;
            sptr[start + tidx] = val;

            __syncthreads();
        }

        val = binop(val, s_tmp[tidy]);

        if (inclusive_scan) {
            if (cond) { optr[id] = val; }
        } else {
            if (id == (out.dims[0] - 1)) {
                optr[0] = init;
            } else if (id < (out.dims[0] - 1)) {
                optr[id + 1] = val;
            }
        }
        id += blockDim.x;
        __syncthreads();
    }

    if (!isFinalPass && isLast) { tptr[blockIdx_x] = val; }
}

template<typename To, af_op_t op>
__global__ void scan_first_bcast(Param<To> out, CParam<To> tmp, uint blocks_x,
                                 uint blocks_y, uint lim, bool inclusive_scan) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int zid        = blockIdx.x / blocks_x;
    const int wid        = (blockIdx.y + blockIdx.z * gridDim.y) / blocks_y;
    const int blockIdx_x = blockIdx.x - (blocks_x)*zid;
    const int blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - (blocks_y)*wid;
    const int xid = blockIdx_x * blockDim.x * lim + tidx;
    const int yid = blockIdx_y * blockDim.y + tidy;

    if (blockIdx_x == 0) return;

    bool cond =
        (yid < out.dims[1]) && (zid < out.dims[2]) && (wid < out.dims[3]);
    if (!cond) return;

    To *optr       = out.ptr;
    const To *tptr = tmp.ptr;

    optr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];
    tptr += wid * tmp.strides[3] + zid * tmp.strides[2] + yid * tmp.strides[1];

    common::Binary<To, op> binop;
    To accum = tptr[blockIdx_x - 1];

    // Shift broadcast one step to the right for exclusive scan (#2366)
    int offset = !inclusive_scan;
    for (int k = 0, id = xid + offset; k < lim && id < out.dims[0];
         k++, id += blockDim.x) {
        optr[id] = binop(accum, optr[id]);
    }
}

}  // namespace cuda
}  // namespace arrayfire
