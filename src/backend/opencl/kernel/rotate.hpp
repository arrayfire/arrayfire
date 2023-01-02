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
#include <kernel/config.hpp>
#include <kernel/interp.hpp>
#include <kernel_headers/interp.hpp>
#include <kernel_headers/rotate.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

typedef struct {
    float tmat[6];
} tmat_t;

template<typename T>
using wtype_t = typename std::conditional<std::is_same<T, double>::value,
                                          double, float>::type;

template<typename T>
using vtype_t = typename std::conditional<common::is_complex<T>::value, T,
                                          wtype_t<T>>::type;

template<typename T>
void rotate(Param out, const Param in, const float theta, af_interp_type method,
            int order) {
    using cl::EnqueueArgs;
    using cl::NDRange;
    using std::string;
    using std::vector;
    using BT = typename dtype_traits<T>::base_type;

    constexpr int TX = 16;
    constexpr int TY = 16;
    // Used for batching images
    constexpr int TI = 4;
    constexpr bool isComplex =
        static_cast<af_dtype>(dtype_traits<T>::af_type) == c32 ||
        static_cast<af_dtype>(dtype_traits<T>::af_type) == c64;

    vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(),
        TemplateArg(order),
    };
    ToNumStr<T> toNumStr;
    vector<string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(ZERO, toNumStr(scalar<T>(0))),
        DefineKeyValue(InterpInTy, dtype_traits<T>::getName()),
        DefineKeyValue(InterpValTy, dtype_traits<vtype_t<T>>::getName()),
        DefineKeyValue(InterpPosTy, dtype_traits<wtype_t<BT>>::getName()),
        DefineKeyValue(XDIM, 0),
        DefineKeyValue(YDIM, 1),
        DefineKeyValue(INTERP_ORDER, order),
        DefineKeyValue(IS_CPLX, (isComplex ? 1 : 0)),
    };
    if (isComplex) {
        compileOpts.emplace_back(
            DefineKeyValue(TB, dtype_traits<BT>::getName()));
    }
    compileOpts.emplace_back(getTypeBuildDefinition<T>());
    addInterpEnumOptions(compileOpts);

    auto rotate =
        common::getKernel("rotateKernel", {{interp_cl_src, rotate_cl_src}},
                          tmpltArgs, compileOpts);

    const float c = cos(-theta), s = sin(-theta);
    float tx, ty;
    {
        const float nx = 0.5 * (in.info.dims[0] - 1);
        const float ny = 0.5 * (in.info.dims[1] - 1);
        const float mx = 0.5 * (out.info.dims[0] - 1);
        const float my = 0.5 * (out.info.dims[1] - 1);
        const float sx = (mx * c + my * -s);
        const float sy = (mx * s + my * c);
        tx             = -(sx - nx);
        ty             = -(sy - ny);
    }

    // Rounding error. Anything more than 3 decimal points wont make a diff
    tmat_t t;
    t.tmat[0] = round(c * 1000) / 1000.0f;
    t.tmat[1] = round(-s * 1000) / 1000.0f;
    t.tmat[2] = round(tx * 1000) / 1000.0f;
    t.tmat[3] = round(s * 1000) / 1000.0f;
    t.tmat[4] = round(c * 1000) / 1000.0f;
    t.tmat[5] = round(ty * 1000) / 1000.0f;

    NDRange local(TX, TY, 1);

    int nimages               = in.info.dims[2];
    int nbatches              = in.info.dims[3];
    int global_x              = local[0] * divup(out.info.dims[0], local[0]);
    int global_y              = local[1] * divup(out.info.dims[1], local[1]);
    const int blocksXPerImage = global_x / local[0];
    const int blocksYPerImage = global_y / local[1];

    if (nimages > TI) {
        int tile_images = divup(nimages, TI);
        nimages         = TI;
        global_x        = global_x * tile_images;
    }
    global_y *= nbatches;

    NDRange global(global_x, global_y, 1);

    rotate(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
           *in.data, in.info, t, nimages, nbatches, blocksXPerImage,
           blocksYPerImage, (int)method);

    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
