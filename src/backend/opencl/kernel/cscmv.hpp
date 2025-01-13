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

namespace arrayfire {
namespace opencl {
namespace kernel {
template<typename T>
void cscmv(Param out, const Param &values, const Param &colIdx,
           const Param &rowIdx, const Param &rhs, const T alpha, const T beta,
           bool is_conj) {
    // TODO: rows_per_group limited by register pressure. Find better way to
    // handle this.
    constexpr int threads_per_g = 64;
    constexpr int rows_per_group = 64;

    const bool use_alpha = (alpha != scalar<T>(1.0));
    const bool use_beta  = (beta != scalar<T>(0.0));

    cl::NDRange local(threads_per_g);

    int K        = colIdx.info.dims[0] - 1;
    int M        = out.info.dims[0];

    std::array<TemplateArg, 5> targs = {
        TemplateTypename<T>(),       TemplateArg(use_alpha),
        TemplateArg(is_conj), TemplateArg(rows_per_group),
        TemplateArg(local[0]),
    };
    std::array<std::string, 9> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(USE_ALPHA, use_alpha),
        DefineKeyValue(IS_CONJ, is_conj),
        DefineKeyValue(THREADS, local[0]),
        DefineKeyValue(ROWS_PER_GROUP, rows_per_group),
        DefineKeyValue(IS_CPLX, (iscplx<T>() ? 1 : 0)),
        DefineKeyValue(IS_DBL, (isdbl<T>() ? 1 : 0)),
        DefineKeyValue(IS_LONG, (islong<T>() ? 1 : 0)),
        getTypeBuildDefinition<T>()};

    if(use_beta) {
        std::array<TemplateArg, 4> targs_beta = {
            TemplateTypename<T>(), TemplateArg(is_conj),
            TemplateArg(rows_per_group), TemplateArg(local[0])};
        std::array<std::string, 8> options_beta = {
            DefineKeyValue(T, dtype_traits<T>::getName()),
            DefineKeyValue(IS_CONJ, is_conj),
            DefineKeyValue(THREADS, local[0]),
            DefineKeyValue(ROWS_PER_GROUP, rows_per_group),
            DefineKeyValue(IS_CPLX, (iscplx<T>() ? 1 : 0)),
            DefineKeyValue(IS_DBL, (isdbl<T>() ? 1 : 0)),
            DefineKeyValue(IS_LONG, (islong<T>() ? 1 : 0)),
            getTypeBuildDefinition<T>()};

        int groups_x = divup(M, rows_per_group * threads_per_g);
        cl::NDRange global(local[0] * groups_x, 1);
        auto cscmvBeta = common::getKernel("cscmv_beta", {{cscmv_cl_src}}, targs_beta, options_beta);
        cscmvBeta(cl::EnqueueArgs(getQueue(), global, local), *out.data, M, beta);

    } else {
        getQueue().enqueueFillBuffer(*out.data, 0, 0, M * sizeof(T));
    }

    int groups_x = divup(M, rows_per_group);
    cl::NDRange global(local[0] * groups_x, 1);

    auto cscmvAtomic =
        common::getKernel("cscmv_atomic", {{cscmv_cl_src}}, targs, options);
    cscmvAtomic(cl::EnqueueArgs(getQueue(), global, local), *out.data,
                *values.data, *colIdx.data, *rowIdx.data, K, *rhs.data,
                rhs.info, alpha);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
