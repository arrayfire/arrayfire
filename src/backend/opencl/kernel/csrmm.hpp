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
#include <kernel_headers/csrmm.hpp>
#include <traits.hpp>
#include <af/opencl.h>

#include <string>
#include <vector>

namespace opencl {
namespace kernel {
template<typename T>
void csrmm_nt(Param out, const Param &values, const Param &rowIdx,
              const Param &colIdx, const Param &rhs, const T alpha,
              const T beta) {
    constexpr int MAX_CSRMM_GROUPS = 4096;
    // Using greedy indexing is causing performance issues on many platforms
    // FIXME: Figure out why
    constexpr bool use_greedy = false;

    static const std::string src(csrmm_cl, csrmm_cl_len);

    const bool use_alpha = (alpha != scalar<T>(1.0));
    const bool use_beta  = (beta != scalar<T>(0.0));

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateArg(use_alpha),
        TemplateArg(use_beta),
        TemplateArg(use_greedy),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(USE_ALPHA, use_alpha),
        DefineKeyValue(USE_BETA, use_beta),
        DefineKeyValue(USE_GREEDY, use_greedy),
        DefineValue(THREADS_PER_GROUP),
        DefineKeyValue(IS_CPLX, (af::iscplx<T>() ? 1 : 0)),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    // FIXME: Switch to perf (thread vs block) baesd kernel
    auto csrmm_nt_func = common::findKernel("csrmm_nt", {src}, targs, options);

    cl::NDRange local(THREADS_PER_GROUP, 1);
    int M = rowIdx.info.dims[0] - 1;
    int N = rhs.info.dims[0];

    int groups_x = divup(N, local[0]);
    int groups_y = divup(M, REPEAT);
    groups_y     = std::min(groups_y, MAX_CSRMM_GROUPS);
    cl::NDRange global(local[0] * groups_x, local[1] * groups_y);

    std::vector<int> count(groups_x);
    cl::Buffer *counter = bufferAlloc(count.size() * sizeof(int));
    getQueue().enqueueWriteBuffer(
        *counter, CL_TRUE, 0, count.size() * sizeof(int), (void *)count.data());

    csrmm_nt_func(cl::EnqueueArgs(getQueue(), global, local), *out.data,
                  *values.data, *rowIdx.data, *colIdx.data, M, N, *rhs.data,
                  rhs.info, alpha, beta, *counter);
    bufferFree(counter);
}
}  // namespace kernel
}  // namespace opencl
