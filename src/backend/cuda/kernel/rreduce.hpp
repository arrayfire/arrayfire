/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <backend.hpp>
#include <common/dispatch.hpp>
#include <debug_cuda.hpp>
#include <err_cuda.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <ops.hpp>
#include <memory>
#include "config.hpp"

namespace cuda {
namespace kernel {

template<typename T, af_op_t op, uint dim, bool is_first, uint DIMY>
__global__ static void rreduce_dim_kernel(Param<T> out, uint *olptr,
                                          CParam<T> in, const uint *ilptr,
                                          uint blocks_x, uint blocks_y,
                                          uint offset_dim, CParam<uint> rlen) {
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
    const uint *rlenptr   = rlen.ptr;

    // There is only one element per block for out
    // There are blockDim.y elements per block for in
    // Hence increment ids[dim] just after offseting out and before offsetting
    // in
    optr += ids[3] * out.strides[3] + ids[2] * out.strides[2] +
            ids[1] * out.strides[1] + ids[0];
    olptr += ids[3] * out.strides[3] + ids[2] * out.strides[2] +
             ids[1] * out.strides[1] + ids[0];
    rlenptr += ids[3] * out.strides[3] + ids[2] * out.strides[2] +
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

    T val    = Binary<T, op>::init();
    uint idx = id_dim_in;

    int lim = min(in.dims[dim], *rlenptr);
    if (is_valid && id_dim_in < lim) {
        val = *iptr;
        if (!is_first) idx = *ilptr;
    }

    MinMaxOp<op, T> Op(val, idx);

    const uint id_dim_in_start = id_dim_in + offset_dim * blockDim.y;

    __shared__ T s_val[THREADS_X * DIMY];
    __shared__ uint s_idx[THREADS_X * DIMY];

    for (int id = id_dim_in_start; is_valid && (id < in.dims[dim]);
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

template<typename T, af_op_t op, int dim, bool is_first>
void rreduce_dim_launcher(Param<T> out, uint *olptr, CParam<T> in,
                          const uint *ilptr, const uint threads_y,
                          const dim_t blocks_dim[4], CParam<uint> rlen) {
    dim3 threads(THREADS_X, threads_y);

    dim3 blocks(blocks_dim[0] * blocks_dim[2], blocks_dim[1] * blocks_dim[3]);

    const int maxBlocksY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    blocks.z = divup(blocks.y, maxBlocksY);
    blocks.y = divup(blocks.y, blocks.z);

    switch (threads_y) {
        case 8:
            CUDA_LAUNCH((rreduce_dim_kernel<T, op, dim, is_first, 8>), blocks,
                        threads, out, olptr, in, ilptr, blocks_dim[0],
                        blocks_dim[1], blocks_dim[dim], rlen);
            break;
        case 4:
            CUDA_LAUNCH((rreduce_dim_kernel<T, op, dim, is_first, 4>), blocks,
                        threads, out, olptr, in, ilptr, blocks_dim[0],
                        blocks_dim[1], blocks_dim[dim], rlen);
            break;
        case 2:
            CUDA_LAUNCH((rreduce_dim_kernel<T, op, dim, is_first, 2>), blocks,
                        threads, out, olptr, in, ilptr, blocks_dim[0],
                        blocks_dim[1], blocks_dim[dim], rlen);
            break;
        case 1:
            CUDA_LAUNCH((rreduce_dim_kernel<T, op, dim, is_first, 1>), blocks,
                        threads, out, olptr, in, ilptr, blocks_dim[0],
                        blocks_dim[1], blocks_dim[dim], rlen);
            break;
    }

    POST_LAUNCH_CHECK();
}

template<typename T, af_op_t op, int dim>
void rreduce_dim(Param<T> out, uint *olptr, CParam<T> in, CParam<uint> rlen) {
    uint threads_y = std::min(THREADS_Y, nextpow2(in.dims[dim]));
    uint threads_x = THREADS_X;

    dim_t blocks_dim[] = {divup(in.dims[0], threads_x), in.dims[1], in.dims[2],
                          in.dims[3]};

    blocks_dim[dim] = divup(in.dims[dim], threads_y * REPEAT);

    Param<T> tmp = out;
    uint *tlptr  = olptr;
    uptr<T> tmp_alloc;
    uptr<uint> tlptr_alloc;

    if (blocks_dim[dim] > 1) {
        int tmp_elements = 1;
        tmp.dims[dim]    = blocks_dim[dim];

        for (int k = 0; k < 4; k++) tmp_elements *= tmp.dims[k];
        tmp_alloc   = memAlloc<T>(tmp_elements);
        tlptr_alloc = memAlloc<uint>(tmp_elements);
        tmp.ptr     = tmp_alloc.get();
        tlptr       = tlptr_alloc.get();

        for (int k = dim + 1; k < 4; k++) tmp.strides[k] *= blocks_dim[dim];
    }

    rreduce_dim_launcher<T, op, dim, true>(tmp, tlptr, in, NULL, threads_y,
                                           blocks_dim, rlen);

    if (blocks_dim[dim] > 1) {
        blocks_dim[dim] = 1;

        rreduce_dim_launcher<T, op, dim, false>(out, olptr, tmp, tlptr,
                                                threads_y, blocks_dim, rlen);
    }
}

template<typename T, af_op_t op, bool is_first, uint DIMX>
__global__ static void rreduce_first_kernel(Param<T> out, uint *olptr,
                                            CParam<T> in, const uint *ilptr,
                                            uint blocks_x, uint blocks_y,
                                            uint repeat, CParam<uint> rlen) {
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
    const uint *rlenptr   = rlen.ptr;

    iptr    += wid * in.strides[3]  + zid * in.strides[2]  + yid * in.strides[1];
    optr    += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];
    rlenptr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];

    if (!is_first)
        ilptr +=
            wid * in.strides[3] + zid * in.strides[2] + yid * in.strides[1];
    olptr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];

    if (yid >= in.dims[1] || zid >= in.dims[2] || wid >= in.dims[3]) return;

    int lim = min((int)(xid + repeat * DIMX), min(*rlenptr, in.dims[0]));

    compute_t<T> val = Binary<compute_t<T>, op>::init();
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


template<typename T, af_op_t op, bool is_first>
void rreduce_first_launcher(Param<T> out, uint *olptr, CParam<T> in,
                            const uint *ilptr, const uint blocks_x,
                            const uint blocks_y, const uint threads_x, CParam<uint> rlen) {
    dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
    dim3 blocks(blocks_x * in.dims[2], blocks_y * in.dims[3]);
    const int maxBlocksY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    blocks.z = divup(blocks.y, maxBlocksY);
    blocks.y = divup(blocks.y, blocks.z);

    uint repeat = divup(in.dims[0], (blocks_x * threads_x));

    switch (threads_x) {
        case 32:
            CUDA_LAUNCH((rreduce_first_kernel<T, op, is_first, 32>), blocks,
                        threads, out, olptr, in, ilptr, blocks_x, blocks_y,
                        repeat, rlen);
            break;
        case 64:
            CUDA_LAUNCH((rreduce_first_kernel<T, op, is_first, 64>), blocks,
                        threads, out, olptr, in, ilptr, blocks_x, blocks_y,
                        repeat, rlen);
            break;
        case 128:
            CUDA_LAUNCH((rreduce_first_kernel<T, op, is_first, 128>), blocks,
                        threads, out, olptr, in, ilptr, blocks_x, blocks_y,
                        repeat, rlen);
            break;
        case 256:
            CUDA_LAUNCH((rreduce_first_kernel<T, op, is_first, 256>), blocks,
                        threads, out, olptr, in, ilptr, blocks_x, blocks_y,
                        repeat, rlen);
            break;
    }

    POST_LAUNCH_CHECK();
}

template<typename T, af_op_t op>
void rreduce_first(Param<T> out, uint *olptr, CParam<T> in, CParam<uint> rlen) {
    uint threads_x = nextpow2(std::max(32u, (uint)in.dims[0]));
    threads_x      = std::min(threads_x, THREADS_PER_BLOCK);
    uint threads_y = THREADS_PER_BLOCK / threads_x;

    uint blocks_x = divup(in.dims[0], threads_x * REPEAT);
    uint blocks_y = divup(in.dims[1], threads_y);

    Param<T> tmp = out;
    uint *tlptr  = olptr;
    uptr<T> tmp_alloc;
    uptr<uint> tlptr_alloc;
    if (blocks_x > 1) {
        auto elements = blocks_x * in.dims[1] * in.dims[2] * in.dims[3];
        tmp_alloc     = memAlloc<T>(elements);
        tlptr_alloc   = memAlloc<uint>(elements);
        tmp.ptr       = tmp_alloc.get();
        tlptr         = tlptr_alloc.get();

        tmp.dims[0] = blocks_x;
        for (int k = 1; k < 4; k++) tmp.strides[k] *= blocks_x;
    }

    rreduce_first_launcher<T, op, true>(tmp, tlptr, in, NULL, blocks_x,
                                        blocks_y, threads_x, rlen);

    if (blocks_x > 1) {
        rreduce_first_launcher<T, op, false>(out, olptr, tmp, tlptr, 1,
                                             blocks_y, threads_x, rlen);
    }
}

template<typename T, af_op_t op>
void rreduce(Param<T> out, uint *olptr, CParam<T> in, int dim, CParam<uint> rlen) {
    switch (dim) {
        case 0: return rreduce_first<T, op>(out, olptr, in, rlen);
        case 1: return rreduce_dim<T, op, 1>(out, olptr, in, rlen);
        case 2: return rreduce_dim<T, op, 2>(out, olptr, in, rlen);
        case 3: return rreduce_dim<T, op, 3>(out, olptr, in, rlen);
    }
}

}  // namespace kernel
}  // namespace cuda
