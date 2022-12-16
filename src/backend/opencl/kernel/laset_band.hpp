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
#include <kernel_headers/laset_band.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

#if 0  // Needs to be enabled when unmqr2 is enabled
static const int NB = 64;
template<int num>
const char *laset_band_name() { return "laset_none"; }
template<> const char *laset_band_name<0>() { return "laset_band_lower"; }
template<> const char *laset_band_name<1>() { return "laset_band_upper"; }

template<typename T, int uplo>
void laset_band(int m, int  n, int k,
                T offdiag, T diag,
                cl_mem dA, size_t dA_offset, magma_int_t ldda)
{
    static const std::string src(laset_band_cl, laset_band_cl_len);

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(), TemplateArg(uplo),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineValue(NB),
        DefineKeyValue(IS_CPLX, static_cast<int>(iscplx<T>())),
        getTypeBuildDefinition<T>()
    };

    auto lasetBandOp = common::getKernel(laset_band_name<uplo>(), {src}, targs, options);

    int threads = 1;
    int groups = 1;

    if (uplo == 0) {
        threads = std::min(k, m);
        groups = (std::min(m, n) - 1) / NB + 1;
    } else {
        threads = std::min(k, n);
        groups = (std::min(m+k-1, n) - 1) / NB + 1;
    }

    cl::NDRange local(threads, 1);
    cl::NDRange global(threads * groups, 1);

    lasetBandOp(cl::EnqueueArgs(getQueue(), global, local), m, n, offdiag, diag, dA, dA_offset, ldda);
}
#endi

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
