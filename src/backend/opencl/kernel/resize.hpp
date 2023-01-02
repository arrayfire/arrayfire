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
#include <common/complex.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/resize.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T>
using wtype_t = typename std::conditional<std::is_same<T, double>::value,
                                          double, float>::type;

template<typename T>
using vtype_t = typename std::conditional<common::is_complex<T>::value, T,
                                          wtype_t<T>>::type;

template<typename T>
void resize(Param out, const Param in, const af_interp_type method) {
    using BT = typename dtype_traits<T>::base_type;

    constexpr int RESIZE_TX = 16;
    constexpr int RESIZE_TY = 16;
    constexpr bool IsComplex =
        std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value;

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateArg(method),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(VT, dtype_traits<vtype_t<T>>::getName()),
        DefineKeyValue(WT, dtype_traits<wtype_t<BT>>::getName()),
        DefineKeyValue(CPLX, (IsComplex ? 1 : 0)),
    };
    if (IsComplex) {
        options.emplace_back(DefineKeyValue(TB, dtype_traits<BT>::getName()));
    }
    options.emplace_back(getTypeBuildDefinition<T>());

    switch (method) {
        case AF_INTERP_NEAREST:
            options.emplace_back(DefineKeyValue(INTERP, "NEAREST"));
            break;
        case AF_INTERP_BILINEAR:
            options.emplace_back(DefineKeyValue(INTERP, "BILINEAR"));
            break;
        case AF_INTERP_LOWER:
            options.emplace_back(DefineKeyValue(INTERP, "LOWER"));
            break;
        default: break;
    }

    auto resizeOp =
        common::getKernel("resize_kernel", {{resize_cl_src}}, targs, options);

    cl::NDRange local(RESIZE_TX, RESIZE_TY, 1);

    int blocksPerMatX = divup(out.info.dims[0], local[0]);
    int blocksPerMatY = divup(out.info.dims[1], local[1]);
    cl::NDRange global(local[0] * blocksPerMatX * in.info.dims[2],
                       local[1] * blocksPerMatY * in.info.dims[3], 1);

    double xd = (double)in.info.dims[0] / (double)out.info.dims[0];
    double yd = (double)in.info.dims[1] / (double)out.info.dims[1];

    float xf = (float)xd, yf = (float)yd;

    resizeOp(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
             *in.data, in.info, blocksPerMatX, blocksPerMatY, xf, yf);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
