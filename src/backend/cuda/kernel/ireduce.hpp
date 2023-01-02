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
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <memory.hpp>
#include <minmax_op.hpp>
#include <nvrtc_kernel_headers/ireduce_cuh.hpp>
#include "config.hpp"

#include <memory>

namespace arrayfire {
namespace cuda {
namespace kernel {

template<typename T, af_op_t op, int dim, bool is_first>
void ireduce_dim_launcher(Param<T> out, uint *olptr, CParam<T> in,
                          const uint *ilptr, const uint threads_y,
                          const dim_t blocks_dim[4], CParam<uint> rlen) {
    dim3 threads(THREADS_X, threads_y);

    dim3 blocks(blocks_dim[0] * blocks_dim[2], blocks_dim[1] * blocks_dim[3]);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    auto ireduceDim = common::getKernel(
        "arrayfire::cuda::ireduceDim", {{ireduce_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(op), TemplateArg(dim),
                     TemplateArg(is_first), TemplateArg(threads_y)),
        {{DefineValue(THREADS_X)}});

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    ireduceDim(qArgs, out, olptr, in, ilptr, blocks_dim[0], blocks_dim[1],
               blocks_dim[dim], rlen);

    POST_LAUNCH_CHECK();
}

template<typename T, af_op_t op, int dim>
void ireduce_dim(Param<T> out, uint *olptr, CParam<T> in, CParam<uint> rlen) {
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

    ireduce_dim_launcher<T, op, dim, true>(tmp, tlptr, in, NULL, threads_y,
                                           blocks_dim, rlen);

    if (blocks_dim[dim] > 1) {
        blocks_dim[dim] = 1;

        ireduce_dim_launcher<T, op, dim, false>(out, olptr, tmp, tlptr,
                                                threads_y, blocks_dim, rlen);
    }
}

template<typename T, af_op_t op, bool is_first>
void ireduce_first_launcher(Param<T> out, uint *olptr, CParam<T> in,
                            const uint *ilptr, const uint blocks_x,
                            const uint blocks_y, const uint threads_x,
                            CParam<uint> rlen) {
    dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
    dim3 blocks(blocks_x * in.dims[2], blocks_y * in.dims[3]);
    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    uint repeat = divup(in.dims[0], (blocks_x * threads_x));

    // threads_x can take values 32, 64, 128, 256
    auto ireduceFirst = common::getKernel(
        "arrayfire::cuda::ireduceFirst", {{ireduce_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(op),
                     TemplateArg(is_first), TemplateArg(threads_x)),
        {{DefineValue(THREADS_PER_BLOCK)}});

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    ireduceFirst(qArgs, out, olptr, in, ilptr, blocks_x, blocks_y, repeat,
                 rlen);
    POST_LAUNCH_CHECK();
}

template<typename T, af_op_t op>
void ireduce_first(Param<T> out, uint *olptr, CParam<T> in, CParam<uint> rlen) {
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

    ireduce_first_launcher<T, op, true>(tmp, tlptr, in, NULL, blocks_x,
                                        blocks_y, threads_x, rlen);

    if (blocks_x > 1) {
        ireduce_first_launcher<T, op, false>(out, olptr, tmp, tlptr, 1,
                                             blocks_y, threads_x, rlen);
    }
}

template<typename T, af_op_t op>
void ireduce(Param<T> out, uint *olptr, CParam<T> in, int dim,
             CParam<uint> rlen) {
    switch (dim) {
        case 0: return ireduce_first<T, op>(out, olptr, in, rlen);
        case 1: return ireduce_dim<T, op, 1>(out, olptr, in, rlen);
        case 2: return ireduce_dim<T, op, 2>(out, olptr, in, rlen);
        case 3: return ireduce_dim<T, op, 3>(out, olptr, in, rlen);
    }
}

template<typename T, af_op_t op>
T ireduce_all(uint *idx, CParam<T> in) {
    using std::unique_ptr;
    int in_elements = in.dims[0] * in.dims[1] * in.dims[2] * in.dims[3];

    // FIXME: Use better heuristics to get to the optimum number
    if (in_elements > 4096) {
        bool is_linear = (in.strides[0] == 1);
        for (int k = 1; k < 4; k++) {
            is_linear &=
                (in.strides[k] == (in.strides[k - 1] * in.dims[k - 1]));
        }

        if (is_linear) {
            in.dims[0] = in_elements;
            for (int k = 1; k < 4; k++) {
                in.dims[k]    = 1;
                in.strides[k] = in_elements;
            }
        }

        uint threads_x = nextpow2(std::max(32u, (uint)in.dims[0]));
        threads_x      = std::min(threads_x, THREADS_PER_BLOCK);
        uint threads_y = THREADS_PER_BLOCK / threads_x;

        Param<T> tmp;
        uint *tlptr;

        uint blocks_x = divup(in.dims[0], threads_x * REPEAT);
        uint blocks_y = divup(in.dims[1], threads_y);

        tmp.dims[0]    = blocks_x;
        tmp.strides[0] = 1;

        for (int k = 1; k < 4; k++) {
            tmp.dims[k]    = in.dims[k];
            tmp.strides[k] = tmp.dims[k - 1] * tmp.strides[k - 1];
        }

        int tmp_elements = tmp.strides[3] * tmp.dims[3];

        // TODO: Use scoped_ptr
        auto tmp_alloc   = memAlloc<T>(tmp_elements);
        auto tlptr_alloc = memAlloc<uint>(tmp_elements);
        tmp.ptr          = tmp_alloc.get();
        tlptr            = tlptr_alloc.get();
        af::dim4 emptysz(0);
        CParam<uint> rlen(nullptr, emptysz.get(), emptysz.get());
        ireduce_first_launcher<T, op, true>(tmp, tlptr, in, NULL, blocks_x,
                                            blocks_y, threads_x, rlen);

        unique_ptr<T[]> h_ptr(new T[tmp_elements]);
        unique_ptr<uint[]> h_lptr(new uint[tmp_elements]);
        T *h_ptr_raw     = h_ptr.get();
        uint *h_lptr_raw = h_lptr.get();

        CUDA_CHECK(cudaMemcpyAsync(h_ptr_raw, tmp.ptr, tmp_elements * sizeof(T),
                                   cudaMemcpyDeviceToHost, getActiveStream()));
        CUDA_CHECK(cudaMemcpyAsync(h_lptr_raw, tlptr,
                                   tmp_elements * sizeof(uint),
                                   cudaMemcpyDeviceToHost, getActiveStream()));
        CUDA_CHECK(cudaStreamSynchronize(getActiveStream()));

        if (!is_linear) {
            // Converting n-d index into a linear index
            // in is of size   [   dims0, dims1, dims2, dims3]
            // tidx is of size [blocks_x, dims1, dims2, dims3]
            // i / blocks_x gives you the batch number "N"
            // "N * dims0 + i" gives the linear index
            for (int i = 0; i < tmp_elements; i++) {
                h_lptr_raw[i] += (i / blocks_x) * in.dims[0];
            }
        }

        MinMaxOp<op, T> Op(h_ptr_raw[0], h_lptr_raw[0]);

        for (int i = 1; i < tmp_elements; i++) {
            Op(h_ptr_raw[i], h_lptr_raw[i]);
        }

        *idx = Op.m_idx;
        return Op.m_val;
    } else {
        unique_ptr<T[]> h_ptr(new T[in_elements]);
        T *h_ptr_raw = h_ptr.get();
        CUDA_CHECK(cudaMemcpyAsync(h_ptr_raw, in.ptr, in_elements * sizeof(T),
                                   cudaMemcpyDeviceToHost, getActiveStream()));
        CUDA_CHECK(cudaStreamSynchronize(getActiveStream()));

        MinMaxOp<op, T> Op(h_ptr_raw[0], 0);
        for (int i = 1; i < in_elements; i++) { Op(h_ptr_raw[i], i); }

        *idx = Op.m_idx;
        return Op.m_val;
    }
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
