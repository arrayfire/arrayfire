/*******************************************************
 * Copyright (c) 2014, ArrayFire
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
#include "config.hpp"
#include "scan_first.hpp"

namespace cuda {
namespace kernel {

template <typename T>
__global__ static void get_out_idx(uint *optr, CParam<uint> otmp,
                                   CParam<uint> rtmp, CParam<T> in,
                                   uint blocks_x, uint blocks_y, uint lim) {
    const uint tidx = threadIdx.x;
    const uint tidy = threadIdx.y;

    const uint zid        = blockIdx.x / blocks_x;
    const uint wid        = (blockIdx.y + blockIdx.z * gridDim.y) / blocks_y;
    const uint blockIdx_x = blockIdx.x - (blocks_x)*zid;
    const uint blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - (blocks_y)*wid;
    const uint xid = blockIdx_x * blockDim.x * lim + tidx;
    const uint yid = blockIdx_y * blockDim.y + tidy;

    const uint *otptr = otmp.ptr;
    const uint *rtptr = rtmp.ptr;
    const T *iptr     = in.ptr;

    const uint off =
        wid * otmp.strides[3] + zid * otmp.strides[2] + yid * otmp.strides[1];
    const uint bid = wid * rtmp.strides[3] + zid * rtmp.strides[2] +
                     yid * rtmp.strides[1] + blockIdx_x;

    otptr +=
        wid * otmp.strides[3] + zid * otmp.strides[2] + yid * otmp.strides[1];
    iptr += wid * in.strides[3] + zid * in.strides[2] + yid * in.strides[1];

    bool cond =
        (yid < otmp.dims[1]) && (zid < otmp.dims[2]) && (wid < otmp.dims[3]);
    T zero = scalar<T>(0);

    if (!cond) return;

    uint accum = (bid == 0) ? 0 : rtptr[bid - 1];

    for (uint k = 0, id = xid; k < lim && id < otmp.dims[0];
         k++, id += blockDim.x) {
        uint idx                            = otptr[id] + accum;
        if (iptr[id] != zero) optr[idx - 1] = (off + id);
    }
}

template <typename T>
static void where(Param<uint> &out, CParam<T> in) {
    uint threads_x = nextpow2(std::max(32u, (uint)in.dims[0]));
    threads_x      = std::min(threads_x, THREADS_PER_BLOCK);
    uint threads_y = THREADS_PER_BLOCK / threads_x;

    uint blocks_x = divup(in.dims[0], threads_x * REPEAT);
    uint blocks_y = divup(in.dims[1], threads_y);

    Param<uint> rtmp;
    Param<uint> otmp;
    rtmp.dims[0]    = blocks_x;
    otmp.dims[0]    = in.dims[0];
    rtmp.strides[0] = 1;
    otmp.strides[0] = 1;

    for (int k = 1; k < 4; k++) {
        rtmp.dims[k]    = in.dims[k];
        rtmp.strides[k] = rtmp.strides[k - 1] * rtmp.dims[k - 1];

        otmp.dims[k]    = in.dims[k];
        otmp.strides[k] = otmp.strides[k - 1] * otmp.dims[k - 1];
    }

    int rtmp_elements = rtmp.strides[3] * rtmp.dims[3];
    int otmp_elements = otmp.strides[3] * otmp.dims[3];
    auto rtmp_alloc   = memAlloc<uint>(rtmp_elements);
    auto otmp_alloc   = memAlloc<uint>(otmp_elements);
    rtmp.ptr          = rtmp_alloc.get();
    otmp.ptr          = otmp_alloc.get();

    scan_first_launcher<T, uint, af_notzero_t, false, true>(
        otmp, rtmp, in, blocks_x, blocks_y, threads_x);

    // Linearize the dimensions and perform scan
    Param<uint> ltmp = rtmp;
    ltmp.dims[0]     = rtmp_elements;
    for (int k = 1; k < 4; k++) {
        ltmp.dims[k]    = 1;
        ltmp.strides[k] = rtmp_elements;
    }

    scan_first<uint, uint, af_add_t, true>(ltmp, ltmp);

    // Get output size and allocate output
    uint total;
    CUDA_CHECK(cudaMemcpyAsync(&total, rtmp.ptr + rtmp_elements - 1,
                               sizeof(uint), cudaMemcpyDeviceToHost,
                               cuda::getActiveStream()));
    CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));

    auto out_alloc = memAlloc<uint>(total);
    out.ptr        = out_alloc.get();

    out.dims[0]    = total;
    out.strides[0] = 1;
    for (int k = 1; k < 4; k++) {
        out.dims[k]    = 1;
        out.strides[k] = total;
    }

    dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
    dim3 blocks(blocks_x * in.dims[2], blocks_y * in.dims[3]);

    uint lim = divup(otmp.dims[0], (threads_x * blocks_x));

    const int maxBlocksY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    blocks.z = divup(blocks.y, maxBlocksY);
    blocks.y = divup(blocks.y, blocks.z);

    CUDA_LAUNCH((get_out_idx<T>), blocks, threads, out.ptr, otmp, rtmp, in,
                blocks_x, blocks_y, lim);
    POST_LAUNCH_CHECK();

    out_alloc.release();
}
}  // namespace kernel
}  // namespace cuda
