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
#include <kernel/config.hpp>
#include <kernel/interp.hpp>
#include <kernel_headers/approx1.hpp>
#include <kernel_headers/approx2.hpp>
#include <kernel_headers/interp.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename Ty, typename Tp>
auto genCompileOptions(const int order, const int xdim, const int ydim = -1) {
    constexpr bool isComplex =
        static_cast<af_dtype>(dtype_traits<Ty>::af_type) == c32 ||
        static_cast<af_dtype>(dtype_traits<Ty>::af_type) == c64;

    ToNumStr<Ty> toNumStr;

    std::vector<std::string> compileOpts = {
        DefineKeyValue(Ty, dtype_traits<Ty>::getName()),
        DefineKeyValue(Tp, dtype_traits<Tp>::getName()),
        DefineKeyValue(InterpInTy, dtype_traits<Ty>::getName()),
        DefineKeyValue(InterpValTy, dtype_traits<Ty>::getName()),
        DefineKeyValue(InterpPosTy, dtype_traits<Tp>::getName()),
        DefineKeyValue(ZERO, toNumStr(scalar<Ty>(0))),
        DefineKeyValue(XDIM, xdim),
        DefineKeyValue(INTERP_ORDER, order),
        DefineKeyValue(IS_CPLX, (isComplex ? 1 : 0)),
    };
    if (ydim != -1) { compileOpts.emplace_back(DefineKeyValue(YDIM, ydim)); }
    compileOpts.emplace_back(getTypeBuildDefinition<Ty>());
    addInterpEnumOptions(compileOpts);

    return compileOpts;
}

template<typename Ty, typename Tp>
void approx1(Param yo, const Param yi, const Param xo, const int xdim,
             const Tp xi_beg, const Tp xi_step, const float offGrid,
             const af_interp_type method, const int order) {
    using cl::EnqueueArgs;
    using cl::NDRange;
    using std::string;
    using std::vector;

    constexpr int THREADS = 256;

    vector<TemplateArg> tmpltArgs = {
        TemplateTypename<Ty>(),
        TemplateTypename<Tp>(),
        TemplateArg(xdim),
        TemplateArg(order),
    };
    auto compileOpts = genCompileOptions<Ty, Tp>(order, xdim);

    auto approx1 = common::getKernel(
        "approx1", {{interp_cl_src, approx1_cl_src}}, tmpltArgs, compileOpts);

    NDRange local(THREADS, 1, 1);
    dim_t blocksPerMat = divup(yo.info.dims[0], local[0]);
    NDRange global(blocksPerMat * local[0] * yo.info.dims[1],
                   yo.info.dims[2] * yo.info.dims[3] * local[1]);

    // Passing bools to opencl kernels is not allowed
    bool batch =
        !(xo.info.dims[1] == 1 && xo.info.dims[2] == 1 && xo.info.dims[3] == 1);

    approx1(EnqueueArgs(getQueue(), global, local), *yo.data, yo.info, *yi.data,
            yi.info, *xo.data, xo.info, xi_beg, Tp(1) / xi_step,
            scalar<Ty>(offGrid), (int)blocksPerMat, (int)batch, (int)method);
    CL_DEBUG_FINISH(getQueue());
}

template<typename Ty, typename Tp>
void approx2(Param zo, const Param zi, const Param xo, const int xdim,
             const Tp &xi_beg, const Tp &xi_step, const Param yo,
             const int ydim, const Tp &yi_beg, const Tp &yi_step,
             const float offGrid, const af_interp_type method,
             const int order) {
    using cl::EnqueueArgs;
    using cl::NDRange;
    using std::string;
    using std::vector;

    constexpr int TX = 16;
    constexpr int TY = 16;

    vector<TemplateArg> tmpltArgs = {
        TemplateTypename<Ty>(), TemplateTypename<Tp>(), TemplateArg(xdim),
        TemplateArg(ydim),      TemplateArg(order),
    };
    auto compileOpts = genCompileOptions<Ty, Tp>(order, xdim, ydim);

    auto approx2 = common::getKernel(
        "approx2", {{interp_cl_src, approx2_cl_src}}, tmpltArgs, compileOpts);

    NDRange local(TX, TY, 1);
    dim_t blocksPerMatX = divup(zo.info.dims[0], local[0]);
    dim_t blocksPerMatY = divup(zo.info.dims[1], local[1]);
    NDRange global(blocksPerMatX * local[0] * zo.info.dims[2],
                   blocksPerMatY * local[1] * zo.info.dims[3], 1);

    // Passing bools to opencl kernels is not allowed
    bool batch = !(xo.info.dims[2] == 1 && xo.info.dims[3] == 1);

    approx2(EnqueueArgs(getQueue(), global, local), *zo.data, zo.info, *zi.data,
            zi.info, *xo.data, xo.info, *yo.data, yo.info, xi_beg,
            Tp(1) / xi_step, yi_beg, Tp(1) / yi_step, scalar<Ty>(offGrid),
            static_cast<int>(blocksPerMatX), static_cast<int>(blocksPerMatY),
            static_cast<int>(batch), static_cast<int>(method));
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
