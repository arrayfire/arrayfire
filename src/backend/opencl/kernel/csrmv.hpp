/*******************************************************
 * Copyright (c) 2016, ArrayFire
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
#include <kernel/config.hpp>
#include <kernel/reduce.hpp>
#include <kernel/scan_dim.hpp>
#include <kernel/scan_first.hpp>
#include <kernel_headers/csrmv.hpp>
#include <traits.hpp>
#include <af/opencl.h>

#include <string>
#include <vector>

namespace opencl {
namespace kernel {
template<typename T>
void csrmv(Param out, const Param &values, const Param &rowIdx,
           const Param &colIdx, const Param &rhs, const T alpha, const T beta) {
    constexpr int MAX_CSRMV_GROUPS = 4096;
    // Using greedy indexing is causing performance issues on many platforms
    // FIXME: Figure out why
    constexpr bool use_greedy = false;
    // FIXME: Find a better number based on average non zeros per row
    constexpr int threads = 64;

    const bool use_alpha = (alpha != scalar<T>(1.0));
    const bool use_beta  = (beta != scalar<T>(0.0));

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),   TemplateArg(use_alpha), TemplateArg(use_beta),
        TemplateArg(use_greedy), TemplateArg(threads),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(USE_ALPHA, use_alpha),
        DefineKeyValue(USE_BETA, use_beta),
        DefineKeyValue(USE_GREEDY, use_greedy),
        DefineKeyValue(THREADS, threads),
        DefineKeyValue(IS_CPLX, (af::iscplx<T>() ? 1 : 0)),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto csrmvThread =
        common::getKernel("csrmv_thread", {csrmv_cl_src}, targs, options);
    auto csrmvBlock =
        common::getKernel("csrmv_block", {csrmv_cl_src}, targs, options);

    int count           = 0;
    cl::Buffer *counter = bufferAlloc(sizeof(int));
    getQueue().enqueueWriteBuffer(*counter, CL_TRUE, 0, sizeof(int),
                                  (void *)&count);

    // TODO: Figure out the proper way to choose either csrmv_thread or
    // csrmv_block
    bool is_csrmv_block = true;
    auto csrmv          = is_csrmv_block ? csrmvBlock : csrmvThread;

    cl::NDRange local(is_csrmv_block ? threads : THREADS_PER_GROUP, 1);
    int M = rowIdx.info.dims[0] - 1;

    int groups_x =
        is_csrmv_block ? divup(M, REPEAT) : divup(M, REPEAT * local[0]);
    groups_x = std::min(groups_x, MAX_CSRMV_GROUPS);
    cl::NDRange global(local[0] * groups_x, 1);

    csrmv(cl::EnqueueArgs(getQueue(), global, local), *out.data, *values.data,
          *rowIdx.data, *colIdx.data, M, *rhs.data, rhs.info, alpha, beta,
          *counter);
    CL_DEBUG_FINISH(getQueue());
    bufferFree(counter);
}
}  // namespace kernel
}  // namespace opencl
