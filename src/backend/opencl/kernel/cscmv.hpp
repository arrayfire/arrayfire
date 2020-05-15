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
#include <kernel_headers/cscmv.hpp>
#include <traits.hpp>
#include <af/opencl.h>

#include <string>

namespace opencl {
namespace kernel {
template<typename T>
void cscmv(Param out, const Param &values, const Param &colIdx,
           const Param &rowIdx, const Param &rhs, const T alpha, const T beta,
           bool is_conj) {
    constexpr int threads = 256;
    // TODO: rows_per_group limited by register pressure. Find better way to
    // handle this.
    constexpr int rows_per_group = 64;

    static const std::string src(cscmv_cl, cscmv_cl_len);

    const bool use_alpha = (alpha != scalar<T>(1.0));
    const bool use_beta  = (beta != scalar<T>(0.0));

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),       TemplateArg(use_alpha),
        TemplateArg(use_beta),       TemplateArg(is_conj),
        TemplateArg(rows_per_group), TemplateArg(threads),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(USE_ALPHA, use_alpha),
        DefineKeyValue(USE_BETA, use_beta),
        DefineKeyValue(IS_CONJ, is_conj),
        DefineKeyValue(THREADS, threads),
        DefineKeyValue(ROWS_PER_GROUP, rows_per_group),
        DefineKeyValue(IS_CPLX, (af::iscplx<T>() ? 1 : 0)),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto cscmvBlock = common::findKernel("cscmv_block", {src}, targs, options);

    cl::NDRange local(threads);
    int K        = colIdx.info.dims[0] - 1;
    int M        = out.info.dims[0];
    int groups_x = divup(M, rows_per_group);
    cl::NDRange global(local[0] * groups_x, 1);

    cscmvBlock(cl::EnqueueArgs(getQueue(), global, local), *out.data,
               *values.data, *colIdx.data, *rowIdx.data, M, K, *rhs.data,
               rhs.info, alpha, beta);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
