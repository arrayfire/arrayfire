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
#include <kernel/config.hpp>
#include <memory.hpp>
#include <nvrtc_kernel_headers/scan_dim_by_key_cuh.hpp>
#include <optypes.hpp>
#include <traits.hpp>

#include <algorithm>

namespace arrayfire {
namespace cuda {
namespace kernel {

template<typename Ti, typename Tk, typename To, af_op_t op>
static void scan_dim_nonfinal_launcher(Param<To> out, Param<To> tmp,
                                       Param<char> tflg, Param<int> tlid,
                                       CParam<Ti> in, CParam<Tk> key,
                                       const int dim, const uint threads_y,
                                       const dim_t blocks_all[4],
                                       bool inclusive_scan) {
    auto scanbykey_dim_nonfinal = common::getKernel(
        "arrayfire::cuda::scanbykey_dim_nonfinal", {{scan_dim_by_key_cuh_src}},
        TemplateArgs(TemplateTypename<Ti>(), TemplateTypename<Tk>(),
                     TemplateTypename<To>(), TemplateArg(op)),
        {{DefineValue(THREADS_X), DefineKeyValue(DIMY, threads_y)}});

    dim3 threads(THREADS_X, threads_y);

    dim3 blocks(blocks_all[0] * blocks_all[2], blocks_all[1] * blocks_all[3]);

    uint lim = divup(out.dims[dim], (threads_y * blocks_all[dim]));

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    scanbykey_dim_nonfinal(qArgs, out, tmp, tflg, tlid, in, key, dim,
                           blocks_all[0], blocks_all[1], lim, inclusive_scan);
    POST_LAUNCH_CHECK();
}

template<typename Ti, typename Tk, typename To, af_op_t op>
static void scan_dim_final_launcher(Param<To> out, CParam<Ti> in,
                                    CParam<Tk> key, const int dim,
                                    const uint threads_y,
                                    const dim_t blocks_all[4],
                                    bool calculateFlags, bool inclusive_scan) {
    auto scanbykey_dim_final = common::getKernel(
        "arrayfire::cuda::scanbykey_dim_final", {{scan_dim_by_key_cuh_src}},
        TemplateArgs(TemplateTypename<Ti>(), TemplateTypename<Tk>(),
                     TemplateTypename<To>(), TemplateArg(op)),
        {{DefineValue(THREADS_X), DefineKeyValue(DIMY, threads_y)}});

    dim3 threads(THREADS_X, threads_y);

    dim3 blocks(blocks_all[0] * blocks_all[2], blocks_all[1] * blocks_all[3]);

    uint lim = divup(out.dims[dim], (threads_y * blocks_all[dim]));

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    scanbykey_dim_final(qArgs, out, in, key, dim, blocks_all[0], blocks_all[1],
                        lim, calculateFlags, inclusive_scan);
    POST_LAUNCH_CHECK();
}

template<typename To, af_op_t op>
static void bcast_dim_launcher(Param<To> out, CParam<To> tmp, Param<int> tlid,
                               const int dim, const uint threads_y,
                               const dim_t blocks_all[4]) {
    auto scanbykey_dim_bcast = common::getKernel(
        "arrayfire::cuda::scanbykey_dim_bcast", {{scan_dim_by_key_cuh_src}},
        TemplateArgs(TemplateTypename<To>(), TemplateArg(op)));
    dim3 threads(THREADS_X, threads_y);
    dim3 blocks(blocks_all[0] * blocks_all[2], blocks_all[1] * blocks_all[3]);

    uint lim = divup(out.dims[dim], (threads_y * blocks_all[dim]));

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    scanbykey_dim_bcast(qArgs, out, tmp, tlid, dim, blocks_all[0],
                        blocks_all[1], blocks_all[dim], lim);
    POST_LAUNCH_CHECK();
}

template<typename Ti, typename Tk, typename To, af_op_t op>
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
        for (int k = 1; k < 4; k++)
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
}  // namespace arrayfire
