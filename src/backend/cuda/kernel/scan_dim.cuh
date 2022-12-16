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

template<typename Ti, typename To, af_op_t op, int dim, bool isFinalPass,
         uint DIMY, bool inclusive_scan>
__global__ void scan_dim(Param<To> out, Param<To> tmp, CParam<Ti> in,
                         uint blocks_x, uint blocks_y, uint blocks_dim,
                         uint lim) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tid  = tidy * THREADS_X + tidx;

    const int zid        = blockIdx.x / blocks_x;
    const int wid        = (blockIdx.y + blockIdx.z * gridDim.y) / blocks_y;
    const int blockIdx_x = blockIdx.x - (blocks_x)*zid;
    const int blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - (blocks_y)*wid;
    const int xid = blockIdx_x * blockDim.x + tidx;
    const int yid = blockIdx_y;  // yid  of output. updated for input later.

    int ids[4] = {xid, yid, zid, wid};

    const Ti *iptr = in.ptr;
    To *optr       = out.ptr;
    To *tptr       = tmp.ptr;

    // There is only one element per block for out
    // There are blockDim.y elements per block for in
    // Hence increment ids[dim] just after offseting out and before offsetting
    // in
    tptr += ids[3] * tmp.strides[3] + ids[2] * tmp.strides[2] +
            ids[1] * tmp.strides[1] + ids[0];
    const int blockIdx_dim = ids[dim];

    ids[dim] = ids[dim] * blockDim.y * lim + tidy;
    optr += ids[3] * out.strides[3] + ids[2] * out.strides[2] +
            ids[1] * out.strides[1] + ids[0];
    iptr += ids[3] * in.strides[3] + ids[2] * in.strides[2] +
            ids[1] * in.strides[1] + ids[0];
    int id_dim        = ids[dim];
    const int out_dim = out.dims[dim];

    bool is_valid = (ids[0] < out.dims[0]) && (ids[1] < out.dims[1]) &&
                    (ids[2] < out.dims[2]) && (ids[3] < out.dims[3]);

    const int ostride_dim = out.strides[dim];
    const int istride_dim = in.strides[dim];

    __shared__ To s_val[THREADS_X * DIMY * 2];
    __shared__ To s_tmp[THREADS_X];
    To *sptr = s_val + tid;

    common::Transform<Ti, To, op> transform;
    common::Binary<To, op> binop;

    const To init = common::Binary<To, op>::init();
    To val        = init;

    const bool isLast = (tidy == (DIMY - 1));

    for (int k = 0; k < lim; k++) {
        if (isLast) s_tmp[tidx] = val;

        bool cond = (is_valid) && (id_dim < out_dim);
        val       = cond ? transform(*iptr) : init;
        *sptr     = val;
        __syncthreads();

        int start = 0;
#pragma unroll
        for (int off = 1; off < DIMY; off *= 2) {
            if (tidy >= off) val = binop(val, sptr[(start - off) * THREADS_X]);
            start                   = DIMY - start;
            sptr[start * THREADS_X] = val;

            __syncthreads();
        }

        val = binop(val, s_tmp[tidx]);
        if (inclusive_scan) {
            if (cond) { *optr = val; }
        } else if (is_valid) {
            if (id_dim == (out_dim - 1)) {
                *(optr - (id_dim * ostride_dim)) = init;
            } else if (id_dim < (out_dim - 1)) {
                *(optr + ostride_dim) = val;
            }
        }
        id_dim += blockDim.y;
        iptr += blockDim.y * istride_dim;
        optr += blockDim.y * ostride_dim;
        __syncthreads();
    }

    if (!isFinalPass && is_valid && (blockIdx_dim < tmp.dims[dim]) && isLast) {
        *tptr = val;
    }
}

template<typename To, af_op_t op, int dim>
__global__ void scan_dim_bcast(Param<To> out, CParam<To> tmp, uint blocks_x,
                               uint blocks_y, uint blocks_dim, uint lim,
                               bool inclusive_scan) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int zid        = blockIdx.x / blocks_x;
    const int wid        = (blockIdx.y + blockIdx.z * gridDim.y) / blocks_y;
    const int blockIdx_x = blockIdx.x - (blocks_x)*zid;
    const int blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - (blocks_y)*wid;
    const int xid = blockIdx_x * blockDim.x + tidx;
    const int yid = blockIdx_y;  // yid  of output. updated for input later.

    int ids[4] = {xid, yid, zid, wid};

    const To *tptr = tmp.ptr;
    To *optr       = out.ptr;

    // There is only one element per block for out
    // There are blockDim.y elements per block for in
    // Hence increment ids[dim] just after offseting out and before offsetting
    // in
    tptr += ids[3] * tmp.strides[3] + ids[2] * tmp.strides[2] +
            ids[1] * tmp.strides[1] + ids[0];
    const int blockIdx_dim = ids[dim];

    ids[dim] = ids[dim] * blockDim.y * lim + tidy;
    optr += ids[3] * out.strides[3] + ids[2] * out.strides[2] +
            ids[1] * out.strides[1] + ids[0];
    const int id_dim  = ids[dim];
    const int out_dim = out.dims[dim];

    // Shift broadcast one step to the right for exclusive scan (#2366)
    int offset = inclusive_scan ? 0 : out.strides[dim];
    optr += offset;

    bool is_valid = (ids[0] < out.dims[0]) && (ids[1] < out.dims[1]) &&
                    (ids[2] < out.dims[2]) && (ids[3] < out.dims[3]);

    if (!is_valid) return;
    if (blockIdx_dim == 0) return;

    To accum = *(tptr - tmp.strides[dim]);

    common::Binary<To, op> binop;
    const int ostride_dim = out.strides[dim];

    for (int k = 0, id = id_dim; is_valid && k < lim && (id < out_dim);
         k++, id += blockDim.y) {
        *optr = binop(*optr, accum);
        optr += blockDim.y * ostride_dim;
    }
}

}  // namespace cuda
}  // namespace arrayfire
