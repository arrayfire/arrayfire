/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <common/Binary.hpp>
#include <common/Transform.hpp>
#include <math.hpp>

namespace arrayfire {
namespace cuda {

template<typename Tk>
__device__ inline char calculate_head_flags(const Tk *kptr, int id,
                                            int previd) {
    return (id == 0) ? 1 : (kptr[id] != kptr[previd]);
}

template<typename Ti, typename Tk, typename To, af_op_t op>
__global__ void scanbykey_first_nonfinal(Param<To> out, Param<To> tmp,
                                         Param<char> tflg, Param<int> tlid,
                                         CParam<Ti> in, CParam<Tk> key,
                                         uint blocks_x, uint blocks_y, uint lim,
                                         bool inclusive_scan) {
    common::Transform<Ti, To, op> transform;
    common::Binary<To, op> binop;
    const To init = common::Binary<To, op>::init();
    To val        = init;

    const int istride         = in.strides[0];
    const int DIMY            = THREADS_PER_BLOCK / DIMX;
    const int SHARED_MEM_SIZE = (2 * DIMX + 1) * (DIMY);
    __shared__ char s_flg[SHARED_MEM_SIZE];
    __shared__ To s_val[SHARED_MEM_SIZE];
    __shared__ char s_ftmp[DIMY];
    __shared__ To s_tmp[DIMY];
    __shared__ int boundaryid[DIMY];

    const int tidx       = threadIdx.x;
    const int tidy       = threadIdx.y;
    const int zid        = blockIdx.x / blocks_x;
    const int wid        = blockIdx.y / blocks_y;
    const int blockIdx_x = blockIdx.x - (blocks_x)*zid;
    const int blockIdx_y = blockIdx.y - (blocks_y)*wid;
    const int xid        = blockIdx_x * blockDim.x * lim + tidx;
    const int yid        = blockIdx_y * blockDim.y + tidy;
    bool cond_yzw =
        (yid < out.dims[1]) && (zid < out.dims[2]) && (wid < out.dims[3]);
    if (!cond_yzw) return;  // retire warps early

    To *sptr    = s_val + tidy * (2 * DIMX + 1);
    char *sfptr = s_flg + tidy * (2 * DIMX + 1);
    int id      = xid;

    const bool isLast = (tidx == (DIMX - 1));
    if (isLast) {
        s_tmp[tidy]      = init;
        s_ftmp[tidy]     = 0;
        boundaryid[tidy] = -1;
    }
    __syncthreads();

    const Ti *iptr = in.ptr;
    const Tk *kptr = key.ptr;
    To *optr       = out.ptr;
    To *tptr       = tmp.ptr;
    char *tfptr    = tflg.ptr;
    int *tiptr     = tlid.ptr;
    iptr += wid * in.strides[3] + zid * in.strides[2] + yid * in.strides[1];
    kptr += wid * key.strides[3] + zid * key.strides[2] + yid * key.strides[1];
    optr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];
    tptr += wid * tmp.strides[3] + zid * tmp.strides[2] + yid * tmp.strides[1];
    tfptr +=
        wid * tflg.strides[3] + zid * tflg.strides[2] + yid * tflg.strides[1];
    tiptr +=
        wid * tlid.strides[3] + zid * tlid.strides[2] + yid * tlid.strides[1];

    char flag = 0;
    for (int k = 0; k < lim; k++) {
        if (id < out.dims[0]) {
            flag = calculate_head_flags(kptr, id, id - 1);
        } else {
            flag = 0;
        }

        // Load val from global in
        if (inclusive_scan) {
            if (id >= out.dims[0]) {
                val = init;
            } else {
                val = transform(iptr[id]);
            }
        } else {
            if ((id == 0) || (id >= out.dims[0]) || flag) {
                val = init;
            } else {
                val = transform(iptr[id - istride]);
            }
        }

        // Add partial result from last iteration before scan operation
        if ((tidx == 0) && (flag == 0)) {
            val  = binop(val, s_tmp[tidy]);
            flag = s_ftmp[tidy];
        }

        // Write to shared memory
        sptr[tidx]  = val;
        sfptr[tidx] = flag;
        __syncthreads();

        // Segmented Scan
        int start = 0;
#pragma unroll
        for (int off = 1; off < DIMX; off *= 2) {
            if (tidx >= off) {
                val  = sfptr[start + tidx]
                           ? val
                           : binop(val, sptr[(start - off) + tidx]);
                flag = sfptr[start + tidx] | sfptr[(start - off) + tidx];
            }
            start               = DIMX - start;
            sptr[start + tidx]  = val;
            sfptr[start + tidx] = flag;

            __syncthreads();
        }

        // Identify segment boundary
        if (tidx == 0) {
            if ((s_ftmp[tidy] == 0) && (sfptr[tidx] == 1)) {
                boundaryid[tidy] = id;
            }
        } else {
            if ((sfptr[tidx - 1] == 0) && (sfptr[tidx] == 1)) {
                boundaryid[tidy] = id;
            }
        }
        __syncthreads();

        if (id < out.dims[0]) optr[id] = val;
        if (isLast) {
            s_tmp[tidy]  = val;
            s_ftmp[tidy] = flag;
        }
        id += blockDim.x;
        __syncthreads();
    }
    if (isLast) {
        tptr[blockIdx_x]  = val;
        tfptr[blockIdx_x] = flag;
        int boundary      = boundaryid[tidy];
        tiptr[blockIdx_x] = (boundary == -1) ? id : boundary;
    }
}

template<typename Ti, typename Tk, typename To, af_op_t op>
__global__ void scanbykey_first_final(Param<To> out, CParam<Ti> in,
                                      CParam<Tk> key, uint blocks_x,
                                      uint blocks_y, uint lim,
                                      bool calculateFlags,
                                      bool inclusive_scan) {
    common::Transform<Ti, To, op> transform;
    common::Binary<To, op> binop;
    const To init = common::Binary<To, op>::init();
    To val        = init;

    const int istride         = in.strides[0];
    const int DIMY            = THREADS_PER_BLOCK / DIMX;
    const int SHARED_MEM_SIZE = (2 * DIMX + 1) * (DIMY);
    __shared__ char s_flg[SHARED_MEM_SIZE];
    __shared__ To s_val[SHARED_MEM_SIZE];
    __shared__ char s_ftmp[DIMY];
    __shared__ To s_tmp[DIMY];

    const int tidx       = threadIdx.x;
    const int tidy       = threadIdx.y;
    const int zid        = blockIdx.x / blocks_x;
    const int wid        = blockIdx.y / blocks_y;
    const int blockIdx_x = blockIdx.x - (blocks_x)*zid;
    const int blockIdx_y = blockIdx.y - (blocks_y)*wid;
    const int xid        = blockIdx_x * blockDim.x * lim + tidx;
    const int yid        = blockIdx_y * blockDim.y + tidy;
    bool cond_yzw =
        (yid < out.dims[1]) && (zid < out.dims[2]) && (wid < out.dims[3]);
    if (!cond_yzw) return;  // retire warps early

    To *sptr    = s_val + tidy * (2 * DIMX + 1);
    char *sfptr = s_flg + tidy * (2 * DIMX + 1);
    int id      = xid;

    const bool isLast = (tidx == (DIMX - 1));
    if (isLast) {
        s_tmp[tidy]  = init;
        s_ftmp[tidy] = 0;
    }
    __syncthreads();

    const Ti *iptr = in.ptr;
    const Tk *kptr = key.ptr;
    To *optr       = out.ptr;
    iptr += wid * in.strides[3] + zid * in.strides[2] + yid * in.strides[1];
    kptr += wid * key.strides[3] + zid * key.strides[2] + yid * key.strides[1];
    optr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];

    for (int k = 0; k < lim; k++) {
        char flag = 0;
        if (calculateFlags) {
            if (id < out.dims[0]) {
                flag = calculate_head_flags(kptr, id, id - key.strides[0]);
            }
        } else {
            flag = kptr[id];
        }

        // Load val from global in
        if (inclusive_scan) {
            if (id >= out.dims[0]) {
                val = init;
            } else {
                val = transform(iptr[id]);
            }
        } else {
            if ((id == 0) || (id >= out.dims[0]) || flag) {
                val = init;
            } else {
                val = transform(iptr[id - istride]);
            }
        }

        // Add partial result from last iteration before scan operation
        if ((tidx == 0) && (flag == 0)) {
            val  = binop(val, s_tmp[tidy]);
            flag = flag | s_ftmp[tidy];
        }

        // Write to shared memory
        sptr[tidx]  = val;
        sfptr[tidx] = flag;
        __syncthreads();

        // Segmented Scan
        int start = 0;
#pragma unroll
        for (int off = 1; off < DIMX; off *= 2) {
            if (tidx >= off) {
                val  = sfptr[start + tidx]
                           ? val
                           : binop(val, sptr[(start - off) + tidx]);
                flag = sfptr[start + tidx] | sfptr[(start - off) + tidx];
            }
            start               = DIMX - start;
            sptr[start + tidx]  = val;
            sfptr[start + tidx] = flag;

            __syncthreads();
        }

        if (id < out.dims[0]) optr[id] = val;
        if (isLast) {
            s_tmp[tidy]  = val;
            s_ftmp[tidy] = flag;
        }
        id += blockDim.x;
        __syncthreads();
    }
}

template<typename To, af_op_t op>
__global__ void scanbykey_first_bcast(Param<To> out, Param<To> tmp,
                                      Param<int> tlid, uint blocks_x,
                                      uint blocks_y, uint lim) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int zid        = blockIdx.x / blocks_x;
    const int wid        = blockIdx.y / blocks_y;
    const int blockIdx_x = blockIdx.x - (blocks_x)*zid;
    const int blockIdx_y = blockIdx.y - (blocks_y)*wid;
    const int xid        = blockIdx_x * blockDim.x * lim + tidx;
    const int yid        = blockIdx_y * blockDim.y + tidy;

    if (blockIdx_x != 0) {
        bool cond =
            (yid < out.dims[1]) && (zid < out.dims[2]) && (wid < out.dims[3]);
        if (cond) {
            To *optr        = out.ptr;
            const To *tptr  = tmp.ptr;
            const int *iptr = tlid.ptr;

            optr += wid * out.strides[3] + zid * out.strides[2] +
                    yid * out.strides[1];
            tptr += wid * tmp.strides[3] + zid * tmp.strides[2] +
                    yid * tmp.strides[1];
            iptr += wid * tlid.strides[3] + zid * tlid.strides[2] +
                    yid * tlid.strides[1];

            common::Binary<To, op> binop;
            int boundary = iptr[blockIdx_x];
            To accum     = tptr[blockIdx_x - 1];

            for (int k = 0, id = xid;
                 k < lim && id < out.dims[0] && id < boundary;
                 k++, id += blockDim.x) {
                optr[id] = binop(accum, optr[id]);
            }
        }
    }
}

}  // namespace cuda
}  // namespace arrayfire
