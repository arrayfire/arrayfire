/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <blas.hpp>
#include <cast.hpp>
#include <common/half.hpp>
#include <common/indexing_helpers.hpp>
#include <common/unique_handle.hpp>
#include <convolve.hpp>
#include <err_cuda.hpp>
#include <kernel/convolve.hpp>
#include <platform.hpp>
#include <reorder.hpp>
#include <transpose.hpp>
#include <unwrap.hpp>
#include <wrap.hpp>
#include <af/dim4.hpp>
#include <type_traits>

using af::dim4;
using common::flip;
using common::half;
using common::make_handle;
using common::unique_handle;
using std::conditional;
using std::is_same;

namespace cuda {

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
    dim4 oDims = dim4(outputWidth, outputHeight, fDims[3], sDims[3]);

    const bool retCols = false;
    Array<T> unwrapped =
        unwrap(signal, fDims[0], fDims[1], stride[0], stride[1], padding[0],
               padding[1], dilation[0], dilation[1], retCols);

    unwrapped  = reorder(unwrapped, dim4(1, 2, 0, 3));
    dim4 uDims = unwrapped.dims();
    unwrapped.modDims(dim4(uDims[0] * uDims[1], uDims[2] * uDims[3]));

    Array<T> collapsedFilter = filter;

    collapsedFilter = flip(collapsedFilter, {1, 1, 0, 0});
    collapsedFilter.modDims(dim4(fDims[0] * fDims[1] * fDims[2], fDims[3]));

    T alpha        = scalar<T>(1.0);
    T beta         = scalar<T>(0.0);
    const int Mdim = 1;
    const int Ndim = 1;
    Array<T> res   = createEmptyArray<T>(
        dim4(unwrapped.dims()[Mdim], collapsedFilter.dims()[Ndim],
             unwrapped.dims()[2], unwrapped.dims()[3]));
    gemm(res, AF_MAT_TRANS, AF_MAT_NONE, &alpha, unwrapped, collapsedFilter,
         &beta);
    res.modDims(dim4(outputWidth, outputHeight, signal.dims()[3],
                     collapsedFilter.dims()[1]));
    Array<T> out = reorder(res, dim4(0, 1, 3, 2));

    return out;
}

template<typename T>
Array<T> convolve2(Array<T> const &signal, Array<T> const &filter,
                   const dim4 stride, const dim4 padding, const dim4 dilation) {
    return convolve2_unwrap<T>(signal, filter, stride, padding, dilation);
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

    Array<T> collapsed_filter = original_filter;

    collapsed_filter = flip(collapsed_filter, {1, 1, 0, 0});
    collapsed_filter.modDims(dim4(fDims[0] * fDims[1] * fDims[2], fDims[3]));

    Array<T> collapsed_gradient = incoming_gradient;
    collapsed_gradient          = reorder(collapsed_gradient, dim4(0, 1, 3, 2));
    collapsed_gradient.modDims(dim4(cDims[0] * cDims[1] * cDims[3], cDims[2]));

    T alpha        = scalar<T>(1.0);
    T beta         = scalar<T>(0.0);
    const int Mdim = 0;
    const int Ndim = 0;
    Array<T> res   = createEmptyArray<T>(
        dim4(collapsed_gradient.dims()[Mdim], collapsed_filter.dims()[Ndim],
             collapsed_gradient.dims()[3], collapsed_gradient.dims()[3]));
    gemm(res, AF_MAT_NONE, AF_MAT_TRANS, &alpha, collapsed_gradient,
         collapsed_filter, &beta);
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
    const dim4 sDims = original_signal.dims();
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

    T alpha        = scalar<T>(1.0);
    T beta         = scalar<T>(0.0);
    const int Mdim = 0;
    const int Ndim = 1;
    Array<T> res   = createEmptyArray<T>(
        dim4(unwrapped.dims()[Mdim], collapsed_gradient.dims()[Ndim],
             unwrapped.dims()[2], unwrapped.dims()[3]));
    gemm(res, AF_MAT_NONE, AF_MAT_NONE, &alpha, unwrapped, collapsed_gradient,
         &beta);
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

}  // namespace cuda
