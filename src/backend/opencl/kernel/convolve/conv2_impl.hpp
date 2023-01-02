/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/kernel_cache.hpp>
#include <kernel/convolve/conv_common.hpp>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T, typename aT>
void conv2Helper(const conv_kparam_t& param, Param out, const Param signal,
                 const Param filter, const bool expand) {
    using cl::EnqueueArgs;
    using cl::NDRange;
    using std::string;
    using std::vector;

    constexpr bool IsComplex =
        std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value;

    const int f0 = filter.info.dims[0];
    const int f1 = filter.info.dims[1];
    const size_t LOC_SIZE =
        (THREADS_X + 2 * (f0 - 1)) * (THREADS_Y + 2 * (f1 - 1));

    vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(), TemplateTypename<aT>(), TemplateArg(expand),
        TemplateArg(f0),       TemplateArg(f1),
    };
    vector<string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(Ti, dtype_traits<T>::getName()),
        DefineKeyValue(To, dtype_traits<aT>::getName()),
        DefineKeyValue(accType, dtype_traits<aT>::getName()),
        DefineKeyValue(RANK, 2),
        DefineKeyValue(FLEN0, f0),
        DefineKeyValue(FLEN1, f1),
        DefineKeyValue(EXPAND, (expand ? 1 : 0)),
        DefineKeyValue(C_SIZE, LOC_SIZE),
        DefineKeyFromStr(binOpName<af_mul_t>()),
        DefineKeyValue(CPLX, (IsComplex ? 1 : 0)),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto convolve = common::getKernel(
        "convolve", {{ops_cl_src, convolve_cl_src}}, tmpltArgs, compileOpts);

    convolve(EnqueueArgs(getQueue(), param.global, param.local), *out.data,
             out.info, *signal.data, signal.info, *param.impulse, filter.info,
             param.nBBS0, param.nBBS1, param.o[1], param.o[2], param.s[1],
             param.s[2]);
}

template<typename T, typename aT>
void conv2(conv_kparam_t& p, Param& out, const Param& sig, const Param& filt,
           const bool expand) {
    size_t se_size = filt.info.dims[0] * filt.info.dims[1] * sizeof(aT);
    p.impulse      = bufferAlloc(se_size);
    int f0Off      = filt.info.offset;

    for (int b3 = 0; b3 < filt.info.dims[3]; ++b3) {
        int f3Off = b3 * filt.info.strides[3];

        for (int b2 = 0; b2 < filt.info.dims[2]; ++b2) {
            int f2Off = b2 * filt.info.strides[2];

            // FIXME: if the filter array is strided, direct copy of symbols
            // might cause issues
            getQueue().enqueueCopyBuffer(*filt.data, *p.impulse,
                                         (f2Off + f3Off + f0Off) * sizeof(aT),
                                         0, se_size);

            p.o[1] = (p.outHasNoOffset ? 0 : b2);
            p.o[2] = (p.outHasNoOffset ? 0 : b3);
            p.s[1] = (p.inHasNoOffset ? 0 : b2);
            p.s[2] = (p.inHasNoOffset ? 0 : b3);

            conv2Helper<T, aT>(p, out, sig, filt, expand);
        }
    }
}

#define INSTANTIATE(T, accT)                                           \
    template void conv2<T, accT>(conv_kparam_t&, Param&, const Param&, \
                                 const Param&, const bool);

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
