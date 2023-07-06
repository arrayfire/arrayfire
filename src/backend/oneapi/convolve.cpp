/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <blas.hpp>
#include <common/half.hpp>
#include <common/indexing_helpers.hpp>
#include <common/moddims.hpp>
#include <convolve.hpp>
#include <err_oneapi.hpp>
#include <kernel/convolve.hpp>
#include <reorder.hpp>
#include <transpose.hpp>
#include <unwrap.hpp>
#include <wrap.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <vector>

using af::dim4;
using arrayfire::common::flip;
using arrayfire::common::half;
using arrayfire::common::modDims;
using std::vector;

namespace arrayfire {
namespace oneapi {

template<typename T, typename accT>
Array<T> convolve(Array<T> const &signal, Array<accT> const &filter,
                  AF_BATCH_KIND kind, const int rank, const bool expand) {
    const dim4 &sDims = signal.dims();
    const dim4 &fDims = filter.dims();

    dim4 oDims(1);
    if (expand) {
        for (int d = 0; d < AF_MAX_DIMS; ++d) {
            if (kind == AF_BATCH_NONE || kind == AF_BATCH_RHS) {
                oDims[d] = sDims[d] + fDims[d] - 1;
            } else {
                oDims[d] = (d < rank ? sDims[d] + fDims[d] - 1 : sDims[d]);
            }
        }
    } else {
        oDims = sDims;
        if (kind == AF_BATCH_RHS) {
            for (int i = rank; i < AF_MAX_DIMS; ++i) { oDims[i] = fDims[i]; }
        }
    }

    Array<T> out    = createEmptyArray<T>(oDims);
    bool callKernel = true;

    dim_t MCFL2 = kernel::MAX_CONV2_FILTER_LEN;
    dim_t MCFL3 = kernel::MAX_CONV3_FILTER_LEN;
    switch (rank) {
        case 1:
            if (fDims[0] > kernel::MAX_CONV1_FILTER_LEN) { callKernel = false; }
            break;
        case 2:
            if ((fDims[0] * fDims[1]) > (MCFL2 * MCFL2)) { callKernel = false; }
            break;
        case 3:
            if ((fDims[0] * fDims[1] * fDims[2]) > (MCFL3 * MCFL3 * MCFL3)) {
                callKernel = false;
            }
            break;
        default: AF_ERROR("rank only supports values 1-3.", AF_ERR_UNKNOWN);
    }

    if (!callKernel) {
        char errMessage[256];
        snprintf(errMessage, sizeof(errMessage),
                 "\nOneAPI N Dimensional Convolution doesn't support "
                 "%llux%llux%llu kernel\n",
                 fDims[0], fDims[1], fDims[2]);
        ONEAPI_NOT_SUPPORTED(errMessage);
    }

    kernel::convolve_nd<T, accT>(out, signal, filter, kind, rank, expand);

    return out;
}

#define INSTANTIATE(T, accT)                                                   \
    template Array<T> convolve<T, accT>(Array<T> const &, Array<accT> const &, \
                                        AF_BATCH_KIND, const int, const bool);

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
                          const dim4 &stride, const dim4 &padding,
                          const dim4 &dilation) {
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

    unwrapped =
        modDims(unwrapped, dim4(uDims[0] * uDims[1], uDims[2] * uDims[3]));

    Array<T> collapsedFilter = filter;

    collapsedFilter = flip(collapsedFilter, {1, 1, 0, 0});
    collapsedFilter = modDims(collapsedFilter,
                              dim4(fDims[0] * fDims[1] * fDims[2], fDims[3]));

    Array<T> res =
        matmul(unwrapped, collapsedFilter, AF_MAT_TRANS, AF_MAT_NONE);
    res = modDims(res, dim4(outputWidth, outputHeight, signal.dims()[3],
                            collapsedFilter.dims()[1]));
    Array<T> out = reorder(res, dim4(0, 1, 3, 2));

    return out;
}

template<typename T>
Array<T> convolve2(Array<T> const &signal, Array<T> const &filter,
                   const dim4 stride, const dim4 padding, const dim4 dilation) {
    Array<T> out =
        convolve2_unwrap<T>(signal, filter, stride, padding, dilation);
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
                           const Array<T> & /*convolved_output*/,
                           af::dim4 stride, af::dim4 padding,
                           af::dim4 dilation) {
    const dim4 &cDims = incoming_gradient.dims();
    const dim4 &sDims = original_signal.dims();
    const dim4 &fDims = original_filter.dims();

    Array<T> collapsed_filter = original_filter;

    collapsed_filter = flip(collapsed_filter, {1, 1, 0, 0});
    collapsed_filter = modDims(collapsed_filter,
                               dim4(fDims[0] * fDims[1] * fDims[2], fDims[3]));

    Array<T> collapsed_gradient = incoming_gradient;
    collapsed_gradient          = reorder(collapsed_gradient, dim4(0, 1, 3, 2));
    collapsed_gradient          = modDims(
        collapsed_gradient, dim4(cDims[0] * cDims[1] * cDims[3], cDims[2]));

    Array<T> res =
        matmul(collapsed_gradient, collapsed_filter, AF_MAT_NONE, AF_MAT_TRANS);
    res = modDims(res, dim4(res.dims()[0] / sDims[3], sDims[3],
                            fDims[0] * fDims[1], sDims[2]));
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
                             const Array<T> & /*convolved_output*/,
                             af::dim4 stride, af::dim4 padding,
                             af::dim4 dilation) {
    const dim4 &cDims = incoming_gradient.dims();
    const dim4 &fDims = original_filter.dims();

    const bool retCols = false;
    Array<T> unwrapped =
        unwrap(original_signal, fDims[0], fDims[1], stride[0], stride[1],
               padding[0], padding[1], dilation[0], dilation[1], retCols);

    unwrapped  = reorder(unwrapped, dim4(1, 2, 0, 3));
    dim4 uDims = unwrapped.dims();
    unwrapped =
        modDims(unwrapped, dim4(uDims[0] * uDims[1], uDims[2] * uDims[3]));

    Array<T> collapsed_gradient = incoming_gradient;
    collapsed_gradient          = reorder(collapsed_gradient, dim4(0, 1, 3, 2));
    collapsed_gradient          = modDims(
        collapsed_gradient, dim4(cDims[0] * cDims[1] * cDims[3], cDims[2]));

    Array<T> res =
        matmul(unwrapped, collapsed_gradient, AF_MAT_NONE, AF_MAT_NONE);
    res = modDims(res, dim4(fDims[0], fDims[1], fDims[2], fDims[3]));

    auto out = flip(res, {1, 1, 0, 0});
    return out;
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

}  // namespace oneapi
}  // namespace arrayfire
