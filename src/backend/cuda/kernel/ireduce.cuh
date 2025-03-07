/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/Binary.hpp>
#include <minmax_op.hpp>

namespace arrayfire {
namespace cuda {

template<typename T, af_op_t op, uint dim, bool is_first, uint DIMY>
__global__ static void ireduceDim(Param<T> out, uint *olptr, CParam<T> in,
                                  const uint *ilptr, uint blocks_x,
                                  uint blocks_y, uint offset_dim,
                                  CParam<uint> rlen) {
    const uint tidx = threadIdx.x;
    const uint tidy = threadIdx.y;
    const uint tid  = tidy * THREADS_X + tidx;

    const uint zid        = blockIdx.x / blocks_x;
    const uint wid        = (blockIdx.y + blockIdx.z * gridDim.y) / blocks_y;
    const uint blockIdx_x = blockIdx.x - (blocks_x)*zid;
    const uint blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - (blocks_y)*wid;
    const uint xid = blockIdx_x * blockDim.x + tidx;
    const uint yid = blockIdx_y;  // yid  of output. updated for input later.

    uint ids[4] = {xid, yid, zid, wid};

    const T *iptr = in.ptr;
    T *optr       = out.ptr;

    // There is only one element per block for out
    // There are blockDim.y elements per block for in
    // Hence increment ids[dim] just after offseting out and before offsetting
    // in
    bool rlen_valid = (ids[0] < rlen.dims[0]) && (ids[1] < rlen.dims[1]) &&
                      (ids[2] < rlen.dims[2]) && (ids[3] < rlen.dims[3]);
    const uint *rlenptr = (rlen.ptr && rlen_valid)
                              ? rlen.ptr + ids[3] * rlen.strides[3] +
                                    ids[2] * rlen.strides[2] +
                                    ids[1] * rlen.strides[1] + ids[0]
                              : nullptr;

    optr += ids[3] * out.strides[3] + ids[2] * out.strides[2] +
            ids[1] * out.strides[1] + ids[0];
    olptr += ids[3] * out.strides[3] + ids[2] * out.strides[2] +
             ids[1] * out.strides[1] + ids[0];

    const uint blockIdx_dim = ids[dim];

    ids[dim] = ids[dim] * blockDim.y + tidy;
    iptr += ids[3] * in.strides[3] + ids[2] * in.strides[2] +
            ids[1] * in.strides[1] + ids[0];
    if (!is_first)
        ilptr += ids[3] * in.strides[3] + ids[2] * in.strides[2] +
                 ids[1] * in.strides[1] + ids[0];
    const uint id_dim_in = ids[dim];

    const uint istride_dim = in.strides[dim];

    bool is_valid = (ids[0] < in.dims[0]) && (ids[1] < in.dims[1]) &&
                    (ids[2] < in.dims[2]) && (ids[3] < in.dims[3]);

    T val    = common::Binary<T, op>::init();
    uint idx = id_dim_in;

    uint lim = (rlenptr) ? *rlenptr : in.dims[dim];
    lim      = (is_first) ? min((uint)in.dims[dim], lim) : lim;
    bool within_ragged_bounds =
        (is_first) ? (idx < lim)
                   : ((rlenptr) ? ((is_valid) && (*ilptr < lim)) : true);
    if (is_valid && id_dim_in < in.dims[dim] && within_ragged_bounds) {
        val = *iptr;
        if (!is_first) idx = *ilptr;
    }

    MinMaxOp<op, T> Op(val, idx);

    const uint id_dim_in_start = id_dim_in + offset_dim * blockDim.y;

    __shared__ T s_val[THREADS_X * DIMY];
    __shared__ uint s_idx[THREADS_X * DIMY];

    for (int id = id_dim_in_start; is_valid && (id < lim);
         id += offset_dim * blockDim.y) {
        iptr = iptr + offset_dim * blockDim.y * istride_dim;
        if (!is_first) {
            ilptr = ilptr + offset_dim * blockDim.y * istride_dim;
            Op(*iptr, *ilptr);
        } else {
            Op(*iptr, id);
        }
    }

    s_val[tid] = Op.m_val;
    s_idx[tid] = Op.m_idx;

    T *s_vptr    = s_val + tid;
    uint *s_iptr = s_idx + tid;
    __syncthreads();

    if (DIMY == 8) {
        if (tidy < 4) {
            Op(s_vptr[THREADS_X * 4], s_iptr[THREADS_X * 4]);
            *s_vptr = Op.m_val;
            *s_iptr = Op.m_idx;
        }
        __syncthreads();
    }

    if (DIMY >= 4) {
        if (tidy < 2) {
            Op(s_vptr[THREADS_X * 2], s_iptr[THREADS_X * 2]);
            *s_vptr = Op.m_val;
            *s_iptr = Op.m_idx;
        }
        __syncthreads();
    }

    if (DIMY >= 2) {
        if (tidy < 1) {
            Op(s_vptr[THREADS_X * 1], s_iptr[THREADS_X * 1]);
            *s_vptr = Op.m_val;
            *s_iptr = Op.m_idx;
        }
        __syncthreads();
    }

    if (tidy == 0 && is_valid && (blockIdx_dim < out.dims[dim])) {
        *optr  = *s_vptr;
        *olptr = *s_iptr;
    }
}

template<typename T, af_op_t op>
__device__ void warp_reduce(T *s_ptr, uint *s_idx, uint tidx) {
    MinMaxOp<op, T> Op(s_ptr[tidx], s_idx[tidx]);
#pragma unroll
    for (int n = 16; n >= 1; n >>= 1) {
        if (tidx < n) {
            Op(s_ptr[tidx + n], s_idx[tidx + n]);
            s_ptr[tidx] = Op.m_val;
            s_idx[tidx] = Op.m_idx;
        }
        __syncthreads();
    }
}

template<typename T, af_op_t op, bool is_first, uint DIMX>
__global__ static void ireduceFirst(Param<T> out, uint *olptr, CParam<T> in,
                                    const uint *ilptr, uint blocks_x,
                                    uint blocks_y, uint repeat,
                                    CParam<uint> rlen) {
    const uint tidx = threadIdx.x;
    const uint tidy = threadIdx.y;
    const uint tid  = tidy * blockDim.x + tidx;

    const uint zid        = blockIdx.x / blocks_x;
    const uint wid        = (blockIdx.y + blockIdx.z * gridDim.y) / blocks_y;
    const uint blockIdx_x = blockIdx.x - (blocks_x)*zid;
    const uint blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - (blocks_y)*wid;
    const uint xid = blockIdx_x * blockDim.x * repeat + tidx;
    const uint yid = blockIdx_y * blockDim.y + tidy;

    const data_t<T> *iptr = in.ptr;
    data_t<T> *optr       = out.ptr;
    const uint *rlenptr   = (rlen.ptr) ? rlen.ptr + wid * rlen.strides[3] +
                                           zid * rlen.strides[2] +
                                           yid * rlen.strides[1]
                                       : nullptr;

    iptr += wid * in.strides[3] + zid * in.strides[2] + yid * in.strides[1];
    optr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];

    if (!is_first)
        ilptr +=
            wid * in.strides[3] + zid * in.strides[2] + yid * in.strides[1];
    olptr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];

    if (yid >= in.dims[1] || zid >= in.dims[2] || wid >= in.dims[3]) return;

    int minlen = rlenptr ? min(*rlenptr, in.dims[0]) : in.dims[0];
    int lim    = min((int)(xid + repeat * DIMX), minlen);

    compute_t<T> val = common::Binary<compute_t<T>, op>::init();
    uint idx         = xid;

    if (xid < lim) {
        val = static_cast<compute_t<T>>(iptr[xid]);
        if (!is_first) idx = ilptr[xid];
    }

    MinMaxOp<op, compute_t<T>> Op(val, idx);

    __shared__ compute_t<T> s_val[THREADS_PER_BLOCK];
    __shared__ uint s_idx[THREADS_PER_BLOCK];

    for (int id = xid + DIMX; id < lim; id += DIMX) {
        Op(static_cast<compute_t<T>>(iptr[id]), (!is_first) ? ilptr[id] : id);
    }

    s_val[tid] = Op.m_val;
    s_idx[tid] = Op.m_idx;
    __syncthreads();

    compute_t<T> *s_vptr = s_val + tidy * DIMX;
    uint *s_iptr         = s_idx + tidy * DIMX;

    if (DIMX == 256) {
        if (tidx < 128) {
            Op(s_vptr[tidx + 128], s_iptr[tidx + 128]);
            s_vptr[tidx] = Op.m_val;
            s_iptr[tidx] = Op.m_idx;
        }
        __syncthreads();
    }

    if (DIMX >= 128) {
        if (tidx < 64) {
            Op(s_vptr[tidx + 64], s_iptr[tidx + 64]);
            s_vptr[tidx] = Op.m_val;
            s_iptr[tidx] = Op.m_idx;
        }
        __syncthreads();
    }

    if (DIMX >= 64) {
        if (tidx < 32) {
            Op(s_vptr[tidx + 32], s_iptr[tidx + 32]);
            s_vptr[tidx] = Op.m_val;
            s_iptr[tidx] = Op.m_idx;
        }
        __syncthreads();
    }

    warp_reduce<compute_t<T>, op>(s_vptr, s_iptr, tidx);

    if (tidx == 0) {
        optr[blockIdx_x]  = s_vptr[0];
        olptr[blockIdx_x] = s_iptr[0];
    }
}

}  // namespace cuda
}  // namespace arrayfire
