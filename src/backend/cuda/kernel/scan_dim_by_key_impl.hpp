/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <Param.hpp>
#include <backend.hpp>
#include <common/dispatch.hpp>
#include <debug_cuda.hpp>
#include <err_cuda.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <ops.hpp>
#include "config.hpp"

namespace cuda {
namespace kernel {

template <typename Tk>
__device__ inline static char calculate_head_flags_dim(const Tk *kptr, int id,
                                                       int stride) {
    return (id == 0) ? 1 : ((*kptr) != (*(kptr - stride)));
}

template <typename Ti, typename Tk, typename To, af_op_t op, uint DIMY>
__global__ static void scan_dim_nonfinal_kernel(Param<To> out, Param<To> tmp,
                                                Param<char> tflg,
                                                Param<int> tlid, CParam<Ti> in,
                                                CParam<Tk> key, int dim,
                                                uint blocks_x, uint blocks_y,
                                                uint lim, bool inclusive_scan) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tid  = tidy * THREADS_X + tidx;

    const int zid        = blockIdx.x / blocks_x;
    const int wid        = blockIdx.y / blocks_y;
    const int blockIdx_x = blockIdx.x - (blocks_x)*zid;
    const int blockIdx_y = blockIdx.y - (blocks_y)*wid;
    const int xid        = blockIdx_x * blockDim.x + tidx;
    const int yid = blockIdx_y;  // yid  of output. updated for input later.

    int ids[4] = {xid, yid, zid, wid};

    const Ti *iptr = in.ptr;
    const Tk *kptr = key.ptr;
    To *optr       = out.ptr;
    To *tptr       = tmp.ptr;
    char *tfptr    = tflg.ptr;
    int *tiptr     = tlid.ptr;

    // There is only one element per block for out
    // There are blockDim.y elements per block for in
    // Hence increment ids[dim] just after offseting out and before offsetting
    // in
    tptr += ids[3] * tmp.strides[3] + ids[2] * tmp.strides[2] +
            ids[1] * tmp.strides[1] + ids[0];
    tfptr += ids[3] * tflg.strides[3] + ids[2] * tflg.strides[2] +
             ids[1] * tflg.strides[1] + ids[0];
    tiptr += ids[3] * tlid.strides[3] + ids[2] * tlid.strides[2] +
             ids[1] * tlid.strides[1] + ids[0];
    const int blockIdx_dim = ids[dim];

    ids[dim] = ids[dim] * blockDim.y * lim + tidy;
    optr += ids[3] * out.strides[3] + ids[2] * out.strides[2] +
            ids[1] * out.strides[1] + ids[0];
    iptr += ids[3] * in.strides[3] + ids[2] * in.strides[2] +
            ids[1] * in.strides[1] + ids[0];
    kptr += ids[3] * key.strides[3] + ids[2] * key.strides[2] +
            ids[1] * key.strides[1] + ids[0];
    int id_dim        = ids[dim];
    const int out_dim = out.dims[dim];

    bool is_valid = (ids[0] < out.dims[0]) && (ids[1] < out.dims[1]) &&
                    (ids[2] < out.dims[2]) && (ids[3] < out.dims[3]);

    const int ostride_dim = out.strides[dim];
    const int istride_dim = in.strides[dim];

    __shared__ char s_flg[THREADS_X * DIMY * 2];
    __shared__ To s_val[THREADS_X * DIMY * 2];
    __shared__ char s_ftmp[THREADS_X];
    __shared__ To s_tmp[THREADS_X];
    __shared__ int boundaryid[THREADS_X];
    To *sptr    = s_val + tid;
    char *sfptr = s_flg + tid;

    Transform<Ti, To, op> transform;
    Binary<To, op> binop;

    const To init = Binary<To, op>::init();
    To val        = init;

    const bool isLast = (tidy == (DIMY - 1));
    if (isLast) {
        s_tmp[tidx]      = val;
        s_ftmp[tidx]     = 0;
        boundaryid[tidx] = -1;
    }
    __syncthreads();

    char flag = 0;
    for (int k = 0; k < lim; k++) {
        if (id_dim < out_dim) {
            flag = calculate_head_flags_dim(kptr, id_dim, key.strides[dim]);
        } else {
            flag = 0;
        }

        // Load val from global in
        if (inclusive_scan) {
            if (id_dim >= out_dim) {
                val = init;
            } else {
                val = transform(*iptr);
            }
        } else {
            if ((id_dim == 0) || (id_dim >= out_dim) || flag) {
                val = init;
            } else {
                val = transform(*(iptr - istride_dim));
            }
        }

        // Add partial result from last iteration before scan operation
        if ((tidy == 0) && (flag == 0)) {
            val  = binop(val, s_tmp[tidx]);
            flag = s_ftmp[tidx];
        }

        // Write to shared memory
        *sptr  = val;
        *sfptr = flag;
        __syncthreads();

        // Segmented Scan
        int start = 0;
#pragma unroll
        for (int off = 1; off < DIMY; off *= 2) {
            if (tidy >= off) {
                val = sfptr[start * THREADS_X]
                          ? val
                          : binop(val, sptr[(start - off) * THREADS_X]);
                flag =
                    sfptr[start * THREADS_X] | sfptr[(start - off) * THREADS_X];
            }
            start                    = DIMY - start;
            sptr[start * THREADS_X]  = val;
            sfptr[start * THREADS_X] = flag;

            __syncthreads();
        }

        // Identify segment boundary
        if (tidy == 0) {
            if ((s_ftmp[tidx] == 0) && (sfptr[start * THREADS_X] == 1)) {
                boundaryid[tidx] = id_dim;
            }
        } else {
            if ((sfptr[(start - 1) * THREADS_X] == 0) &&
                (sfptr[start * THREADS_X] == 1)) {
                boundaryid[tidx] = id_dim;
            }
        }
        __syncthreads();

        if (is_valid && (id_dim < out_dim)) *optr = val;
        if (isLast) {
            s_tmp[tidx]  = val;
            s_ftmp[tidx] = flag;
        }
        id_dim += blockDim.y;
        kptr += blockDim.y * key.strides[dim];
        iptr += blockDim.y * istride_dim;
        optr += blockDim.y * ostride_dim;
        __syncthreads();
    }

    if (is_valid && (blockIdx_dim < tmp.dims[dim]) && isLast) {
        *tptr        = val;
        *tfptr       = flag;
        int boundary = boundaryid[tidx];
        *tiptr       = (boundary == -1) ? id_dim : boundary;
    }
}

template <typename Ti, typename Tk, typename To, af_op_t op, uint DIMY>
__global__ static void scan_dim_final_kernel(Param<To> out, CParam<Ti> in,
                                             CParam<Tk> key, int dim,
                                             uint blocks_x, uint blocks_y,
                                             uint lim, bool calculateFlags,
                                             bool inclusive_scan) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tid  = tidy * THREADS_X + tidx;

    const int zid        = blockIdx.x / blocks_x;
    const int wid        = blockIdx.y / blocks_y;
    const int blockIdx_x = blockIdx.x - (blocks_x)*zid;
    const int blockIdx_y = blockIdx.y - (blocks_y)*wid;
    const int xid        = blockIdx_x * blockDim.x + tidx;
    const int yid = blockIdx_y;  // yid  of output. updated for input later.

    int ids[4] = {xid, yid, zid, wid};

    const Ti *iptr = in.ptr;
    const Tk *kptr = key.ptr;
    To *optr       = out.ptr;

    // There is only one element per block for out
    // There are blockDim.y elements per block for in
    // Hence increment ids[dim] just after offseting out and before offsetting
    // in

    ids[dim] = ids[dim] * blockDim.y * lim + tidy;
    optr += ids[3] * out.strides[3] + ids[2] * out.strides[2] +
            ids[1] * out.strides[1] + ids[0];
    iptr += ids[3] * in.strides[3] + ids[2] * in.strides[2] +
            ids[1] * in.strides[1] + ids[0];
    kptr += ids[3] * key.strides[3] + ids[2] * key.strides[2] +
            ids[1] * key.strides[1] + ids[0];
    int id_dim        = ids[dim];
    const int out_dim = out.dims[dim];

    bool is_valid = (ids[0] < out.dims[0]) && (ids[1] < out.dims[1]) &&
                    (ids[2] < out.dims[2]) && (ids[3] < out.dims[3]);

    const int ostride_dim = out.strides[dim];
    const int istride_dim = in.strides[dim];

    __shared__ char s_flg[THREADS_X * DIMY * 2];
    __shared__ To s_val[THREADS_X * DIMY * 2];
    __shared__ char s_ftmp[THREADS_X];
    __shared__ To s_tmp[THREADS_X];
    To *sptr    = s_val + tid;
    char *sfptr = s_flg + tid;

    Transform<Ti, To, op> transform;
    Binary<To, op> binop;

    const To init = Binary<To, op>::init();
    To val        = init;

    const bool isLast = (tidy == (DIMY - 1));
    if (isLast) {
        s_tmp[tidx]  = val;
        s_ftmp[tidx] = 0;
    }
    __syncthreads();

    char flag = 0;
    for (int k = 0; k < lim; k++) {
        if (calculateFlags) {
            if (id_dim < out_dim) {
                flag = calculate_head_flags_dim(kptr, id_dim, key.strides[dim]);
            } else {
                flag = 0;
            }
        } else {
            flag = *kptr;
        }

        // Load val from global in
        if (inclusive_scan) {
            if (id_dim >= out_dim) {
                val = init;
            } else {
                val = transform(*iptr);
            }
        } else {
            if ((id_dim == 0) || (id_dim >= out_dim) || flag) {
                val = init;
            } else {
                val = transform(*(iptr - istride_dim));
            }
        }

        // Add partial result from last iteration before scan operation
        if ((tidy == 0) && (flag == 0)) {
            val  = binop(val, s_tmp[tidx]);
            flag = s_ftmp[tidx];
        }

        // Write to shared memory
        *sptr  = val;
        *sfptr = flag;
        __syncthreads();

        // Segmented Scan
        int start = 0;
#pragma unroll
        for (int off = 1; off < DIMY; off *= 2) {
            if (tidy >= off) {
                val = sfptr[start * THREADS_X]
                          ? val
                          : binop(val, sptr[(start - off) * THREADS_X]);
                flag =
                    sfptr[start * THREADS_X] | sfptr[(start - off) * THREADS_X];
            }
            start                    = DIMY - start;
            sptr[start * THREADS_X]  = val;
            sfptr[start * THREADS_X] = flag;

            __syncthreads();
        }

        if (is_valid && (id_dim < out_dim)) *optr = val;
        if (isLast) {
            s_tmp[tidx]  = val;
            s_ftmp[tidx] = flag;
        }
        id_dim += blockDim.y;
        kptr += blockDim.y * key.strides[dim];
        iptr += blockDim.y * istride_dim;
        optr += blockDim.y * ostride_dim;
        __syncthreads();
    }
}

template <typename To, af_op_t op>
__global__ static void bcast_dim_kernel(Param<To> out, CParam<To> tmp,
                                        Param<int> tlid, int dim, uint blocks_x,
                                        uint blocks_y, uint blocks_dim,
                                        uint lim) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int zid        = blockIdx.x / blocks_x;
    const int wid        = blockIdx.y / blocks_y;
    const int blockIdx_x = blockIdx.x - (blocks_x)*zid;
    const int blockIdx_y = blockIdx.y - (blocks_y)*wid;
    const int xid        = blockIdx_x * blockDim.x + tidx;
    const int yid = blockIdx_y;  // yid  of output. updated for input later.

    int ids[4] = {xid, yid, zid, wid};

    const To *tptr  = tmp.ptr;
    To *optr        = out.ptr;
    const int *iptr = tlid.ptr;

    // There is only one element per block for out
    // There are blockDim.y elements per block for in
    // Hence increment ids[dim] just after offseting out and before offsetting
    // in
    tptr += ids[3] * tmp.strides[3] + ids[2] * tmp.strides[2] +
            ids[1] * tmp.strides[1] + ids[0];
    iptr += ids[3] * tlid.strides[3] + ids[2] * tlid.strides[2] +
            ids[1] * tlid.strides[1] + ids[0];
    const int blockIdx_dim = ids[dim];

    ids[dim] = ids[dim] * blockDim.y * lim + tidy;
    optr += ids[3] * out.strides[3] + ids[2] * out.strides[2] +
            ids[1] * out.strides[1] + ids[0];
    const int id_dim = ids[dim];

    bool is_valid = (ids[0] < out.dims[0]) && (ids[1] < out.dims[1]) &&
                    (ids[2] < out.dims[2]) && (ids[3] < out.dims[3]);

    if (!is_valid) return;
    if (blockIdx_dim == 0) return;

    int boundary = *iptr;
    To accum     = *(tptr - tmp.strides[dim]);

    Binary<To, op> binop;
    const int ostride_dim = out.strides[dim];

    for (int k = 0, id = id_dim; is_valid && k < lim && (id < boundary);
         k++, id += blockDim.y) {
        *optr = binop(*optr, accum);
        optr += blockDim.y * ostride_dim;
    }
}

template <typename Ti, typename Tk, typename To, af_op_t op>
static void scan_dim_final_launcher(Param<To> out, CParam<Ti> in,
                                    CParam<Tk> key, const int dim,
                                    const uint threads_y,
                                    const dim_t blocks_all[4],
                                    bool calculateFlags, bool inclusive_scan) {
    dim3 threads(THREADS_X, threads_y);

    dim3 blocks(blocks_all[0] * blocks_all[2], blocks_all[1] * blocks_all[3]);

    uint lim = divup(out.dims[dim], (threads_y * blocks_all[dim]));

    switch (threads_y) {
        case 8:
            CUDA_LAUNCH((scan_dim_final_kernel<Ti, Tk, To, op, 8>), blocks,
                        threads, out, in, key, dim, blocks_all[0],
                        blocks_all[1], lim, calculateFlags, inclusive_scan);
            break;
        case 4:
            CUDA_LAUNCH((scan_dim_final_kernel<Ti, Tk, To, op, 4>), blocks,
                        threads, out, in, key, dim, blocks_all[0],
                        blocks_all[1], lim, calculateFlags, inclusive_scan);
            break;
        case 2:
            CUDA_LAUNCH((scan_dim_final_kernel<Ti, Tk, To, op, 2>), blocks,
                        threads, out, in, key, dim, blocks_all[0],
                        blocks_all[1], lim, calculateFlags, inclusive_scan);
            break;
        case 1:
            CUDA_LAUNCH((scan_dim_final_kernel<Ti, Tk, To, op, 1>), blocks,
                        threads, out, in, key, dim, blocks_all[0],
                        blocks_all[1], lim, calculateFlags, inclusive_scan);
            break;
    }

    POST_LAUNCH_CHECK();
}

template <typename Ti, typename Tk, typename To, af_op_t op>
static void scan_dim_nonfinal_launcher(Param<To> out, Param<To> tmp,
                                       Param<char> tflg, Param<int> tlid,
                                       CParam<Ti> in, CParam<Tk> key,
                                       const int dim, const uint threads_y,
                                       const dim_t blocks_all[4],
                                       bool inclusive_scan) {
    dim3 threads(THREADS_X, threads_y);

    dim3 blocks(blocks_all[0] * blocks_all[2], blocks_all[1] * blocks_all[3]);

    uint lim = divup(out.dims[dim], (threads_y * blocks_all[dim]));

    switch (threads_y) {
        case 8:
            CUDA_LAUNCH((scan_dim_nonfinal_kernel<Ti, Tk, To, op, 8>), blocks,
                        threads, out, tmp, tflg, tlid, in, key, dim,
                        blocks_all[0], blocks_all[1], lim, inclusive_scan);
            break;
        case 4:
            CUDA_LAUNCH((scan_dim_nonfinal_kernel<Ti, Tk, To, op, 4>), blocks,
                        threads, out, tmp, tflg, tlid, in, key, dim,
                        blocks_all[0], blocks_all[1], lim, inclusive_scan);
            break;
        case 2:
            CUDA_LAUNCH((scan_dim_nonfinal_kernel<Ti, Tk, To, op, 2>), blocks,
                        threads, out, tmp, tflg, tlid, in, key, dim,
                        blocks_all[0], blocks_all[1], lim, inclusive_scan);
            break;
        case 1:
            CUDA_LAUNCH((scan_dim_nonfinal_kernel<Ti, Tk, To, op, 1>), blocks,
                        threads, out, tmp, tflg, tlid, in, key, dim,
                        blocks_all[0], blocks_all[1], lim, inclusive_scan);
            break;
    }

    POST_LAUNCH_CHECK();
}

template <typename To, af_op_t op>
static void bcast_dim_launcher(Param<To> out, CParam<To> tmp, Param<int> tlid,
                               const int dim, const uint threads_y,
                               const dim_t blocks_all[4]) {
    dim3 threads(THREADS_X, threads_y);

    dim3 blocks(blocks_all[0] * blocks_all[2], blocks_all[1] * blocks_all[3]);

    uint lim = divup(out.dims[dim], (threads_y * blocks_all[dim]));

    CUDA_LAUNCH((bcast_dim_kernel<To, op>), blocks, threads, out, tmp, tlid,
                dim, blocks_all[0], blocks_all[1], blocks_all[dim], lim);

    POST_LAUNCH_CHECK();
}

template <typename Ti, typename Tk, typename To, af_op_t op>
void scan_dim_by_key(Param<To> out, CParam<Ti> in, CParam<Tk> key, int dim,
                     bool inclusive_scan) {
    uint threads_y = std::min(THREADS_Y, nextpow2(out.dims[dim]));
    uint threads_x = THREADS_X;

    dim_t blocks_all[] = {divup(out.dims[0], threads_x), out.dims[1],
                          out.dims[2], out.dims[3]};

    blocks_all[dim] = divup(out.dims[dim], threads_y * REPEAT);

    if (blocks_all[dim] == 1) {
        scan_dim_final_launcher<Ti, Tk, To, op>(
            out, in, key, dim, threads_y, blocks_all, true, inclusive_scan);

    } else {
        Param<To> tmp = out;
        Param<char> tmpflg;
        Param<int> tmpid;

        tmp.dims[dim]  = blocks_all[dim];
        tmp.strides[0] = 1;
        for (int k         = 1; k < 4; k++)
            tmp.strides[k] = tmp.strides[k - 1] * tmp.dims[k - 1];
        for (int k = 0; k < 4; k++) {
            tmpflg.strides[k] = tmp.strides[k];
            tmpid.strides[k]  = tmp.strides[k];
            tmpflg.dims[k]    = tmp.dims[k];
            tmpid.dims[k]     = tmp.dims[k];
        }

        int tmp_elements  = tmp.strides[3] * tmp.dims[3];
        auto tmp_alloc    = memAlloc<To>(tmp_elements);
        auto tmpflg_alloc = memAlloc<char>(tmp_elements);
        auto tmpid_alloc  = memAlloc<int>(tmp_elements);
        tmp.ptr           = tmp_alloc.get();
        tmpflg.ptr        = tmpflg_alloc.get();
        tmpid.ptr         = tmpid_alloc.get();

        scan_dim_nonfinal_launcher<Ti, Tk, To, op>(out, tmp, tmpflg, tmpid, in,
                                                   key, dim, threads_y,
                                                   blocks_all, inclusive_scan);

        int bdim        = blocks_all[dim];
        blocks_all[dim] = 1;
        scan_dim_final_launcher<To, char, To, op>(
            tmp, tmp, tmpflg, dim, threads_y, blocks_all, false, true);

        blocks_all[dim] = bdim;
        bcast_dim_launcher<To, op>(out, tmp, tmpid, dim, threads_y, blocks_all);
    }
}

}  // namespace kernel

#define INSTANTIATE_SCAN_DIM_BY_KEY(ROp, Ti, Tk, To)           \
    template void scan_dim_by_key<Ti, Tk, To, ROp>(            \
        Param<To> out, CParam<Ti> in, CParam<Tk> key, int dim, \
        bool inclusive_scan);

#define INSTANTIATE_SCAN_DIM_BY_KEY_TYPES(ROp, Tk)         \
    INSTANTIATE_SCAN_DIM_BY_KEY(ROp, float, Tk, float)     \
    INSTANTIATE_SCAN_DIM_BY_KEY(ROp, double, Tk, double)   \
    INSTANTIATE_SCAN_DIM_BY_KEY(ROp, cfloat, Tk, cfloat)   \
    INSTANTIATE_SCAN_DIM_BY_KEY(ROp, cdouble, Tk, cdouble) \
    INSTANTIATE_SCAN_DIM_BY_KEY(ROp, int, Tk, int)         \
    INSTANTIATE_SCAN_DIM_BY_KEY(ROp, uint, Tk, uint)       \
    INSTANTIATE_SCAN_DIM_BY_KEY(ROp, intl, Tk, intl)       \
    INSTANTIATE_SCAN_DIM_BY_KEY(ROp, uintl, Tk, uintl)

#define INSTANTIATE_SCAN_DIM_BY_KEY_OP(ROp)      \
    INSTANTIATE_SCAN_DIM_BY_KEY_TYPES(ROp, int)  \
    INSTANTIATE_SCAN_DIM_BY_KEY_TYPES(ROp, uint) \
    INSTANTIATE_SCAN_DIM_BY_KEY_TYPES(ROp, intl) \
    INSTANTIATE_SCAN_DIM_BY_KEY_TYPES(ROp, uintl)
}  // namespace cuda
