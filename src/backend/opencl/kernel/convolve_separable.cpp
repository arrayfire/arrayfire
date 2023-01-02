/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel_headers/convolve_separable.hpp>

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/err_common.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel/names.hpp>
#include <kernel_headers/ops.hpp>
#include <memory.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T, typename accType>
void convSep(Param out, const Param signal, const Param filter,
             const int conv_dim, const bool expand) {
    if (!(conv_dim == 0 || conv_dim == 1)) {
        AF_ERROR(
            "Separable convolution accepts only 0 or 1 as convolution "
            "dimension",
            AF_ERR_NOT_SUPPORTED);
    }
    constexpr int THREADS_X = 16;
    constexpr int THREADS_Y = 16;
    constexpr bool IsComplex =
        std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value;

    const int fLen       = filter.info.dims[0] * filter.info.dims[1];
    const size_t C0_SIZE = (THREADS_X + 2 * (fLen - 1)) * THREADS_Y;
    const size_t C1_SIZE = (THREADS_Y + 2 * (fLen - 1)) * THREADS_X;
    size_t locSize       = (conv_dim == 0 ? C0_SIZE : C1_SIZE);

    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(), TemplateTypename<accType>(),
        TemplateArg(conv_dim), TemplateArg(expand),
        TemplateArg(fLen),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(Ti, dtype_traits<T>::getName()),
        DefineKeyValue(To, dtype_traits<accType>::getName()),
        DefineKeyValue(accType, dtype_traits<accType>::getName()),
        DefineKeyValue(CONV_DIM, conv_dim),
        DefineKeyValue(EXPAND, (expand ? 1 : 0)),
        DefineKeyValue(FLEN, fLen),
        DefineKeyFromStr(binOpName<af_mul_t>()),
        DefineKeyValue(IS_CPLX, (IsComplex ? 1 : 0)),
        DefineKeyValue(LOCAL_MEM_SIZE, locSize),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto conv =
        common::getKernel("convolve", {{ops_cl_src, convolve_separable_cl_src}},
                          tmpltArgs, compileOpts);

    cl::NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(out.info.dims[0], THREADS_X);
    int blk_y = divup(out.info.dims[1], THREADS_Y);

    cl::NDRange global(blk_x * signal.info.dims[2] * THREADS_X,
                       blk_y * signal.info.dims[3] * THREADS_Y);

    cl::Buffer *mBuff = bufferAlloc(fLen * sizeof(accType));
    // FIX ME: if the filter array is strided, direct might cause issues
    getQueue().enqueueCopyBuffer(*filter.data, *mBuff, 0, 0,
                                 fLen * sizeof(accType));

    conv(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
         *signal.data, signal.info, *mBuff, blk_x, blk_y);
    bufferFree(mBuff);
}

#define INSTANTIATE(T, accT)                                             \
    template void convSep<T, accT>(Param, const Param, const Param filt, \
                                   const int, const bool);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat, cfloat)
INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(uint, float)
INSTANTIATE(int, float)
INSTANTIATE(uchar, float)
INSTANTIATE(char, float)
INSTANTIATE(ushort, float)
INSTANTIATE(short, float)
INSTANTIATE(uintl, float)
INSTANTIATE(intl, float)

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
