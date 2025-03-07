/*******************************************************
 * Copyright (c) 2023, ArrayFire
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
#include <debug_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <af/defines.h>

#include <sycl/sycl.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

// below shared MAX_*_LEN's are calculated based on
// a maximum shared memory configuration of 48KB per block
// considering complex types as well
constexpr int MAX_CONV1_FILTER_LEN = 129;
constexpr int MAX_CONV2_FILTER_LEN = 17;
constexpr int MAX_CONV3_FILTER_LEN = 5;

constexpr int MAX_SCONV_FILTER_LEN = 31;

constexpr int THREADS   = 256;
constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;
constexpr int CUBE_X    = 8;
constexpr int CUBE_Y    = 8;
constexpr int CUBE_Z    = 4;

template<typename aT>
struct conv_kparam_t {
    sycl::range<3> global{0, 0, 0};
    sycl::range<3> local{0, 0, 0};
    size_t loc_size;
    int nBBS0;
    int nBBS1;
    bool outHasNoOffset;
    bool inHasNoOffset;
    bool launchMoreBlocks;
    int o[3];
    int s[3];
    sycl::buffer<aT> *impulse;
};

template<typename T>
T binOp(T lhs, T rhs) {
    return lhs * rhs;
}

template<typename aT>
void prepareKernelArgs(conv_kparam_t<aT> &param, dim_t *oDims,
                       const dim_t *fDims, const int rank) {
    using sycl::range;

    int batchDims[4] = {1, 1, 1, 1};
    for (int i = rank; i < 4; ++i) {
        batchDims[i] = (param.launchMoreBlocks ? 1 : oDims[i]);
    }

    if (rank == 1) {
        param.local    = range<3>{THREADS, 1, 1};
        param.nBBS0    = divup(oDims[0], THREADS);
        param.nBBS1    = batchDims[2];
        param.global   = range<3>(param.nBBS0 * THREADS * batchDims[1],
                                param.nBBS1 * batchDims[3], 1);
        param.loc_size = (THREADS + 2 * (fDims[0] - 1));
    } else if (rank == 2) {
        param.local  = range<3>{THREADS_X, THREADS_Y, 1};
        param.nBBS0  = divup(oDims[0], THREADS_X);
        param.nBBS1  = divup(oDims[1], THREADS_Y);
        param.global = range<3>(param.nBBS0 * THREADS_X * batchDims[2],
                                param.nBBS1 * THREADS_Y * batchDims[3], 1);
    } else if (rank == 3) {
        param.local    = range<3>{CUBE_X, CUBE_Y, CUBE_Z};
        param.nBBS0    = divup(oDims[0], CUBE_X);
        param.nBBS1    = divup(oDims[1], CUBE_Y);
        int blk_z      = divup(oDims[2], CUBE_Z);
        param.global   = range<3>(param.nBBS0 * CUBE_X * batchDims[3],
                                param.nBBS1 * CUBE_Y, blk_z * CUBE_Z);
        param.loc_size = (CUBE_X + 2 * (fDims[0] - 1)) *
                         (CUBE_Y + 2 * (fDims[1] - 1)) *
                         (CUBE_Z + 2 * (fDims[2] - 1));
    }
}

template<typename T>
void memcpyBuffer(sycl::buffer<T, 1> &dest, sycl::buffer<T, 1> &src,
                  const size_t n, const size_t srcOffset) {
    getQueue().submit([&](auto &h) {
        sycl::accessor srcAcc{src, h, sycl::range{n}, sycl::id{srcOffset},
                              sycl::read_only};
        sycl::accessor destAcc{
            dest,         h, sycl::range{n}, sycl::id{0}, sycl::write_only,
            sycl::no_init};
        h.copy(srcAcc, destAcc);
    });
}

#include "convolve1.hpp"
#include "convolve2.hpp"
#include "convolve3.hpp"

template<typename T, typename aT>
void convolve_nd(Param<T> out, const Param<T> signal, const Param<aT> filter,
                 AF_BATCH_KIND kind, const int rank, const bool expand) {
    conv_kparam_t<aT> param;

    for (int i = 0; i < 3; ++i) {
        param.o[i] = 0;
        param.s[i] = 0;
    }
    param.launchMoreBlocks = kind == AF_BATCH_SAME || kind == AF_BATCH_RHS;
    param.outHasNoOffset   = kind == AF_BATCH_LHS || kind == AF_BATCH_NONE;
    param.inHasNoOffset    = kind != AF_BATCH_SAME;

    prepareKernelArgs<aT>(param, out.info.dims, filter.info.dims, rank);

    switch (rank) {
        case 1: conv1<T, aT>(param, out, signal, filter, expand); break;
        case 2: conv2<T, aT>(param, out, signal, filter, expand); break;
        case 3: conv3<T, aT>(param, out, signal, filter, expand); break;
    }

    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
