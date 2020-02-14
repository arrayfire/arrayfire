/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <arith.hpp>
#include <blas.hpp>
#include <common/defines.hpp>
#include <common/half.hpp>
#include <common/indexing_helpers.hpp>
#include <convolve.hpp>
#include <handle.hpp>
#include <kernel/convolve.hpp>
#include <platform.hpp>
#include <reorder.hpp>
#include <transpose.hpp>
#include <unwrap.hpp>
#include <wrap.hpp>
#include <vector>

#include <af/defines.h>
#include <af/dim4.hpp>

using af::dim4;
using common::flip;
using common::half;
using std::vector;

namespace cpu {

template<typename T, typename accT, int baseDim, bool expand>
Array<T> convolve(Array<T> const &signal, Array<accT> const &filter,
                  AF_BATCH_KIND kind) {
    auto sDims = signal.dims();
    auto fDims = filter.dims();

    dim4 oDims(1);
    if (expand) {
        for (dim_t d = 0; d < 4; ++d) {
            if (kind == AF_BATCH_NONE || kind == AF_BATCH_RHS) {
                oDims[d] = sDims[d] + fDims[d] - 1;
            } else {
                oDims[d] = (d < baseDim ? sDims[d] + fDims[d] - 1 : sDims[d]);
            }
        }
    } else {
        oDims = sDims;
        if (kind == AF_BATCH_RHS) {
            for (dim_t i = baseDim; i < 4; ++i) oDims[i] = fDims[i];
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);

    getQueue().enqueue(kernel::convolve_nd<T, accT, baseDim, expand>, out,
                       signal, filter, kind);

    return out;
}

template<typename T, typename accT, bool expand>
Array<T> convolve2(Array<T> const &signal, Array<accT> const &c_filter,
                   Array<accT> const &r_filter) {
    auto sDims = signal.dims();
    dim4 tDims = sDims;
    dim4 oDims = sDims;

    if (expand) {
        auto cfDims = c_filter.dims();
        auto rfDims = r_filter.dims();

        dim_t cflen = (dim_t)cfDims.elements();
        dim_t rflen = (dim_t)rfDims.elements();
        // separable convolve only does AF_BATCH_NONE and standard
        // batch(AF_BATCH_LHS)
        tDims[0] += cflen - 1;
        oDims[0] += cflen - 1;
        oDims[1] += rflen - 1;
    }

    Array<T> out  = createEmptyArray<T>(oDims);
    Array<T> temp = createEmptyArray<T>(tDims);

    getQueue().enqueue(kernel::convolve2<T, accT, expand>, out, signal,
                       c_filter, r_filter, temp);

    return out;
}

#define INSTANTIATE(T, accT)                                                 \
    template Array<T> convolve<T, accT, 1, true>(Array<T> const &signal,     \
                                                 Array<accT> const &filter,  \
                                                 AF_BATCH_KIND kind);        \
    template Array<T> convolve<T, accT, 1, false>(Array<T> const &signal,    \
                                                  Array<accT> const &filter, \
                                                  AF_BATCH_KIND kind);       \
    template Array<T> convolve<T, accT, 2, true>(Array<T> const &signal,     \
                                                 Array<accT> const &filter,  \
                                                 AF_BATCH_KIND kind);        \
    template Array<T> convolve<T, accT, 2, false>(Array<T> const &signal,    \
                                                  Array<accT> const &filter, \
                                                  AF_BATCH_KIND kind);       \
    template Array<T> convolve<T, accT, 3, true>(Array<T> const &signal,     \
                                                 Array<accT> const &filter,  \
                                                 AF_BATCH_KIND kind);        \
    template Array<T> convolve<T, accT, 3, false>(Array<T> const &signal,    \
                                                  Array<accT> const &filter, \
                                                  AF_BATCH_KIND kind);       \
    template Array<T> convolve2<T, accT, true>(Array<T> const &signal,       \
                                               Array<accT> const &c_filter,  \
                                               Array<accT> const &r_filter); \
    template Array<T> convolve2<T, accT, false>(Array<T> const &signal,      \
                                                Array<accT> const &c_filter, \
                                                Array<accT> const &r_filter);

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
#undef INSTANTIATE

template<typename T>
Array<T> convolve2_unwrap(const Array<T> &signal, const Array<T> &filter,
                          const dim4 stride, const dim4 padding,
                          const dim4 dilation) {
    dim4 sDims = signal.dims();
    dim4 fDims = filter.dims();

    dim_t outputWidth =
        1 + (sDims[0] + 2 * padding[0] - (((fDims[0] - 1) * dilation[0]) + 1)) /
                stride[0];
    dim_t outputHeight =
        1 + (sDims[1] + 2 * padding[1] - (((fDims[1] - 1) * dilation[1]) + 1)) /
                stride[1];

    const bool retCols = false;
    Array<T> unwrapped =
        unwrap(signal, fDims[0], fDims[1], stride[0], stride[1], padding[0],
               padding[1], dilation[0], dilation[1], retCols);

    unwrapped  = reorder(unwrapped, dim4(1, 2, 0, 3));
    dim4 uDims = unwrapped.dims();
    unwrapped.modDims(dim4(uDims[0] * uDims[1], uDims[2] * uDims[3]));

    Array<T> collapsedFilter = flip(filter, {1, 1, 0, 0});
    collapsedFilter.modDims(dim4(fDims[0] * fDims[1] * fDims[2], fDims[3]));

    Array<T> res =
        matmul(unwrapped, collapsedFilter, AF_MAT_TRANS, AF_MAT_NONE);
    res.modDims(dim4(outputWidth, outputHeight, signal.dims()[3],
                     collapsedFilter.dims()[1]));
    Array<T> out = reorder(res, dim4(0, 1, 3, 2));

    return out;
}

template<typename T>
Array<T> convolve2(Array<T> const &signal, Array<T> const &filter,
                   const dim4 stride, const dim4 padding, const dim4 dilation) {
    Array<T> out = createEmptyArray<T>(dim4());
    out = convolve2_unwrap<T>(signal, filter, stride, padding, dilation);

    return out;
}

#define INSTANTIATE(T)                                                        \
    template Array<T> convolve2<T>(Array<T> const &signal,                    \
                                   Array<T> const &filter, const dim4 stride, \
                                   const dim4 padding, const dim4 dilation);

INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(half)
#undef INSTANTIATE

template<typename T>
Array<T> conv2DataGradient(const Array<T> &incoming_gradient,
                           const Array<T> &original_signal,
                           const Array<T> &original_filter,
                           const Array<T> &convolved_output, af::dim4 stride,
                           af::dim4 padding, af::dim4 dilation) {
    const dim4 cDims = incoming_gradient.dims();
    const dim4 sDims = original_signal.dims();
    const dim4 fDims = original_filter.dims();

    Array<T> collapsed_filter = flip(original_filter, {1, 1, 0, 0});
    collapsed_filter.modDims(dim4(fDims[0] * fDims[1] * fDims[2], fDims[3]));

    Array<T> collapsed_gradient = incoming_gradient;
    collapsed_gradient          = reorder(collapsed_gradient, dim4(0, 1, 3, 2));
    collapsed_gradient.modDims(dim4(cDims[0] * cDims[1] * cDims[3], cDims[2]));

    Array<T> res =
        matmul(collapsed_gradient, collapsed_filter, AF_MAT_NONE, AF_MAT_TRANS);
    res.modDims(dim4(res.dims()[0] / sDims[3], sDims[3], fDims[0] * fDims[1],
                     sDims[2]));
    res = reorder(res, dim4(0, 2, 3, 1));

    const bool retCols = false;
    res = wrap_dilated(res, sDims[0], sDims[1], fDims[0], fDims[1], stride[0],
                       stride[1], padding[0], padding[1], dilation[0],
                       dilation[1], retCols);

    return res;
}

template<typename T>
Array<T> conv2FilterGradient(const Array<T> &incoming_gradient,
                             const Array<T> &original_signal,
                             const Array<T> &original_filter,
                             const Array<T> &convolved_output, af::dim4 stride,
                             af::dim4 padding, af::dim4 dilation) {
    const dim4 cDims = incoming_gradient.dims();
    const dim4 fDims = original_filter.dims();

    const bool retCols = false;
    Array<T> unwrapped =
        unwrap(original_signal, fDims[0], fDims[1], stride[0], stride[1],
               padding[0], padding[1], dilation[0], dilation[1], retCols);

    unwrapped  = reorder(unwrapped, dim4(1, 2, 0, 3));
    dim4 uDims = unwrapped.dims();
    unwrapped.modDims(dim4(uDims[0] * uDims[1], uDims[2] * uDims[3]));

    Array<T> collapsed_gradient = incoming_gradient;
    collapsed_gradient          = reorder(collapsed_gradient, dim4(0, 1, 3, 2));
    collapsed_gradient.modDims(dim4(cDims[0] * cDims[1] * cDims[3], cDims[2]));

    Array<T> res =
        matmul(unwrapped, collapsed_gradient, AF_MAT_NONE, AF_MAT_NONE);
    res.modDims(dim4(fDims[0], fDims[1], fDims[2], fDims[3]));

    return flip(res, {1, 1, 0, 0});
}

#define INSTANTIATE(T)                                                      \
    template Array<T> conv2DataGradient<T>(                                 \
        Array<T> const &incoming_gradient, Array<T> const &original_signal, \
        Array<T> const &original_filter, Array<T> const &convolved_output,  \
        const dim4 stride, const dim4 padding, const dim4 dilation);        \
    template Array<T> conv2FilterGradient<T>(                               \
        Array<T> const &incoming_gradient, Array<T> const &original_signal, \
        Array<T> const &original_filter, Array<T> const &convolved_output,  \
        const dim4 stride, const dim4 padding, const dim4 dilation);

INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(half)
#undef INSTANTIATE

}  // namespace cpu
