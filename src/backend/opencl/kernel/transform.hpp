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
#include <kernel_headers/transform.hpp>
#include <math.hpp>
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
void transform(Param out, const Param in, const Param tf, bool isInverse,
               bool isPerspective, af_interp_type method, int order) {
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
        TemplateArg(isInverse),
        TemplateArg(isPerspective),
        TemplateArg(order),
    };
    ToNumStr<T> toNumStr;
    vector<string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(INVERSE, (isInverse ? 1 : 0)),
        DefineKeyValue(PERSPECTIVE, (isPerspective ? 1 : 0)),
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

    auto transform = common::getKernel("transformKernel",
                                       {{interp_cl_src, transform_cl_src}},
                                       tmpltArgs, compileOpts);

    const int nImg2 = in.info.dims[2];
    const int nImg3 = in.info.dims[3];
    const int nTfs2 = tf.info.dims[2];
    const int nTfs3 = tf.info.dims[3];

    NDRange local(TX, TY, 1);

    int batchImg2 = 1;
    if (nImg2 != nTfs2) batchImg2 = min(nImg2, TI);

    const int blocksXPerImage = divup(out.info.dims[0], local[0]);
    const int blocksYPerImage = divup(out.info.dims[1], local[1]);

    int global_x = local[0] * blocksXPerImage * (nImg2 / batchImg2);
    int global_y = local[1] * blocksYPerImage * nImg3;
    int global_z = local[2] * max((nTfs2 / nImg2), 1) * max((nTfs3 / nImg3), 1);

    NDRange global(global_x, global_y, global_z);

    transform(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
              *in.data, in.info, *tf.data, tf.info, nImg2, nImg3, nTfs2, nTfs3,
              batchImg2, blocksXPerImage, blocksYPerImage, (int)method);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
