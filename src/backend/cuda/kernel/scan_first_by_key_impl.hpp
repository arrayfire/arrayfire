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
#include <memory.hpp>
#include <nvrtc/cache.hpp>
#include <nvrtc_kernel_headers/scan_first_by_key.hpp>
#include <optypes.hpp>
#include "config.hpp"

namespace cuda {
namespace kernel {

static const std::string ScanFirstByKeySource(scan_first_by_key_cuh,
                                              scan_first_by_key_cuh_len);

template<typename Ti, typename Tk, typename To, af_op_t op>
static void scan_nonfinal_launcher(Param<To> out, Param<To> tmp,
                                   Param<char> tflg, Param<int> tlid,
                                   CParam<Ti> in, CParam<Tk> key,
                                   const uint blocks_x, const uint blocks_y,
                                   const uint threads_x, bool inclusive_scan) {
    // clang-format off
    auto scanNonFinal = getKernel("cuda::scanbykey_first_nonfinal",
            ScanFirstByKeySource,
            {
              TemplateTypename<Ti>(),
              TemplateTypename<Tk>(),
              TemplateTypename<To>(),
              TemplateArg(op)
            },
            {
              DefineValue(THREADS_PER_BLOCK),
              DefineKeyValue(DIMX, threads_x)
            }
            );
    // clang-format on
    dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
    dim3 blocks(blocks_x * out.dims[2], blocks_y * out.dims[3]);

    uint lim = divup(out.dims[0], (threads_x * blocks_x));

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    scanNonFinal(qArgs, out, tmp, tflg, tlid, in, key, blocks_x, blocks_y, lim,
                 inclusive_scan);
    POST_LAUNCH_CHECK();
}

template<typename Ti, typename Tk, typename To, af_op_t op>
static void scan_final_launcher(Param<To> out, CParam<Ti> in, CParam<Tk> key,
                                const uint blocks_x, const uint blocks_y,
                                const uint threads_x, bool calculateFlags,
                                bool inclusive_scan) {
    // clang-format off
    auto scanFinal = getKernel("cuda::scanbykey_first_final",
            ScanFirstByKeySource,
            {
              TemplateTypename<Ti>(),
              TemplateTypename<Tk>(),
              TemplateTypename<To>(),
              TemplateArg(op)
            },
            {
              DefineValue(THREADS_PER_BLOCK),
              DefineKeyValue(DIMX, threads_x)
            }
            );
    // clang-format on
    dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
    dim3 blocks(blocks_x * out.dims[2], blocks_y * out.dims[3]);

    uint lim = divup(out.dims[0], (threads_x * blocks_x));

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    scanFinal(qArgs, out, in, key, blocks_x, blocks_y, lim, calculateFlags,
              inclusive_scan);
    POST_LAUNCH_CHECK();
}

template<typename To, af_op_t op>
static void bcast_first_launcher(Param<To> out, Param<To> tmp, Param<int> tlid,
                                 const dim_t blocks_x, const dim_t blocks_y,
                                 const uint threads_x) {
    // clang-format off
    auto bcastFirst = getKernel("cuda::scanbykey_first_bcast",
            ScanFirstByKeySource,
            {
              TemplateTypename<To>(),
              TemplateArg(op)
            }
            );
    // clang-format on
    dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
    dim3 blocks(blocks_x * out.dims[2], blocks_y * out.dims[3]);
    uint lim = divup(out.dims[0], (threads_x * blocks_x));

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    bcastFirst(qArgs, out, tmp, tlid, blocks_x, blocks_y, lim);
    POST_LAUNCH_CHECK();
}

template<typename Ti, typename Tk, typename To, af_op_t op>
void scan_first_by_key(Param<To> out, CParam<Ti> in, CParam<Tk> key,
                       bool inclusive_scan) {
    uint threads_x = nextpow2(std::max(32u, (uint)out.dims[0]));
    threads_x      = std::min(threads_x, THREADS_PER_BLOCK);
    uint threads_y = THREADS_PER_BLOCK / threads_x;

    uint blocks_x = static_cast<uint>(divup(out.dims[0], threads_x * REPEAT));
    uint blocks_y = static_cast<uint>(divup(out.dims[1], threads_y));

    if (blocks_x == 1) {
        scan_final_launcher<Ti, Tk, To, op>(out, in, key, blocks_x, blocks_y,
                                            threads_x, true, inclusive_scan);

    } else {
        Param<To> tmp = out;
        Param<char> tmpflg;
        Param<int> tmpid;

        tmp.dims[0]       = blocks_x;
        tmpflg.dims[0]    = blocks_x;
        tmpid.dims[0]     = blocks_x;
        tmp.strides[0]    = 1;
        tmpflg.strides[0] = 1;
        tmpid.strides[0]  = 1;
        for (int k = 1; k < 4; k++) {
            tmpflg.dims[k]    = out.dims[k];
            tmpid.dims[k]     = out.dims[k];
            tmp.strides[k]    = tmp.strides[k - 1] * tmp.dims[k - 1];
            tmpflg.strides[k] = tmpflg.strides[k - 1] * tmpflg.dims[k - 1];
            tmpid.strides[k]  = tmpid.strides[k - 1] * tmpid.dims[k - 1];
        }

        int tmp_elements  = tmp.strides[3] * tmp.dims[3];
        auto tmp_alloc    = memAlloc<To>(tmp_elements);
        auto tmpflg_alloc = memAlloc<char>(tmp_elements);
        auto tmpid_alloc  = memAlloc<int>(tmp_elements);
        tmp.ptr           = tmp_alloc.get();
        tmpflg.ptr        = tmpflg_alloc.get();
        tmpid.ptr         = tmpid_alloc.get();

        scan_nonfinal_launcher<Ti, Tk, To, op>(out, tmp, tmpflg, tmpid, in, key,
                                               blocks_x, blocks_y, threads_x,
                                               inclusive_scan);

        scan_final_launcher<To, char, To, op>(tmp, tmp, tmpflg, 1, blocks_y,
                                              threads_x, false, true);

        bcast_first_launcher<To, op>(out, tmp, tmpid, blocks_x, blocks_y,
                                     threads_x);
    }
}
}  // namespace kernel

#define INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, Ti, Tk, To) \
    template void scan_first_by_key<Ti, Tk, To, ROp>(  \
        Param<To> out, CParam<Ti> in, CParam<Tk> key, bool inclusive_scan);

#define INSTANTIATE_SCAN_FIRST_BY_KEY_TYPES(ROp, Tk)         \
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, float, Tk, float)     \
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, double, Tk, double)   \
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, cfloat, Tk, cfloat)   \
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, cdouble, Tk, cdouble) \
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, int, Tk, int)         \
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, uint, Tk, uint)       \
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, intl, Tk, intl)       \
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, uintl, Tk, uintl)

#define INSTANTIATE_SCAN_FIRST_BY_KEY_OP(ROp)      \
    INSTANTIATE_SCAN_FIRST_BY_KEY_TYPES(ROp, int)  \
    INSTANTIATE_SCAN_FIRST_BY_KEY_TYPES(ROp, uint) \
    INSTANTIATE_SCAN_FIRST_BY_KEY_TYPES(ROp, intl) \
    INSTANTIATE_SCAN_FIRST_BY_KEY_TYPES(ROp, uintl)
}  // namespace cuda
