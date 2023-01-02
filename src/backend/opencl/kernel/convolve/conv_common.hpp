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
#include <debug_opencl.hpp>
#include <kernel/names.hpp>
#include <kernel_headers/convolve.hpp>
#include <kernel_headers/ops.hpp>
#include <memory.hpp>
#include <traits.hpp>
#include <types.hpp>
#include <af/defines.h>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

constexpr int THREADS   = 256;
constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;
constexpr int CUBE_X    = 8;
constexpr int CUBE_Y    = 8;
constexpr int CUBE_Z    = 4;

struct conv_kparam_t {
    cl::NDRange global;
    cl::NDRange local;
    size_t loc_size;
    int nBBS0;
    int nBBS1;
    bool outHasNoOffset;
    bool inHasNoOffset;
    bool launchMoreBlocks;
    int o[3];
    int s[3];
    cl::Buffer* impulse;
};

template<typename T>
void prepareKernelArgs(conv_kparam_t& param, dim_t* oDims, const dim_t* fDims,
                       const int rank) {
    using cl::NDRange;

    int batchDims[4] = {1, 1, 1, 1};
    for (int i = rank; i < 4; ++i) {
        batchDims[i] = (param.launchMoreBlocks ? 1 : oDims[i]);
    }

    if (rank == 1) {
        param.local    = NDRange(THREADS, 1);
        param.nBBS0    = divup(oDims[0], THREADS);
        param.nBBS1    = batchDims[2];
        param.global   = NDRange(param.nBBS0 * THREADS * batchDims[1],
                                 param.nBBS1 * batchDims[3]);
        param.loc_size = (THREADS + 2 * (fDims[0] - 1)) * sizeof(T);
    } else if (rank == 2) {
        param.local  = NDRange(THREADS_X, THREADS_Y);
        param.nBBS0  = divup(oDims[0], THREADS_X);
        param.nBBS1  = divup(oDims[1], THREADS_Y);
        param.global = NDRange(param.nBBS0 * THREADS_X * batchDims[2],
                               param.nBBS1 * THREADS_Y * batchDims[3]);
    } else if (rank == 3) {
        param.local    = NDRange(CUBE_X, CUBE_Y, CUBE_Z);
        param.nBBS0    = divup(oDims[0], CUBE_X);
        param.nBBS1    = divup(oDims[1], CUBE_Y);
        int blk_z      = divup(oDims[2], CUBE_Z);
        param.global   = NDRange(param.nBBS0 * CUBE_X * batchDims[3],
                                 param.nBBS1 * CUBE_Y, blk_z * CUBE_Z);
        param.loc_size = (CUBE_X + 2 * (fDims[0] - 1)) *
                         (CUBE_Y + 2 * (fDims[1] - 1)) *
                         (CUBE_Z + 2 * (fDims[2] - 1)) * sizeof(T);
    }
}

template<typename T, typename aT>
void convNHelper(const conv_kparam_t& param, Param& out, const Param& signal,
                 const Param& filter, const int rank, const bool expand) {
    using cl::EnqueueArgs;
    using cl::NDRange;
    using std::string;
    using std::vector;

    constexpr bool IsComplex =
        std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value;

    vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(),
        TemplateTypename<aT>(),
        TemplateArg(rank),
        TemplateArg(expand),
    };
    vector<string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(Ti, dtype_traits<T>::getName()),
        DefineKeyValue(To, dtype_traits<aT>::getName()),
        DefineKeyValue(accType, dtype_traits<aT>::getName()),
        DefineKeyValue(RANK, rank),
        DefineKeyValue(EXPAND, (expand ? 1 : 0)),
        DefineKeyFromStr(binOpName<af_mul_t>()),
        DefineKeyValue(CPLX, (IsComplex ? 1 : 0)),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto convolve = common::getKernel(
        "convolve", {{ops_cl_src, convolve_cl_src}}, tmpltArgs, compileOpts);

    convolve(EnqueueArgs(getQueue(), param.global, param.local), *out.data,
             out.info, *signal.data, signal.info, cl::Local(param.loc_size),
             *param.impulse, filter.info, param.nBBS0, param.nBBS1, param.o[0],
             param.o[1], param.o[2], param.s[0], param.s[1], param.s[2]);
}

template<typename T, typename aT>
void conv1(conv_kparam_t& p, Param& out, const Param& sig, const Param& filt,
           const bool expand);

template<typename T, typename aT>
void conv2(conv_kparam_t& p, Param& out, const Param& sig, const Param& filt,
           const bool expand);

template<typename T, typename aT>
void conv3(conv_kparam_t& p, Param& out, const Param& sig, const Param& filt,
           const bool expand);
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
