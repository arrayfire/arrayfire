/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <convolve.hpp>

#include <arith.hpp>
#include <backend.hpp>
#include <cast.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <fftconvolve.hpp>
#include <handle.hpp>
#include <tile.hpp>
#include <af/data.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/ml.h>
#include <af/signal.h>

#include <cstdio>

using af::dim4;
using common::half;
using detail::arithOp;
using detail::Array;
using detail::cast;
using detail::cdouble;
using detail::cfloat;
using detail::convolve;
using detail::intl;
using detail::uchar;
using detail::uintl;

template<typename T, typename accT, dim_t baseDim, bool expand>
inline static af_array convolve(const af_array &s, const af_array &f,
                                AF_BATCH_KIND kind) {
    return getHandle(convolve<T, accT, baseDim, expand>(
        getArray<T>(s), castArray<accT>(f), kind));
}

template<typename T, typename accT, bool expand>
inline static af_array convolve2(const af_array &s, const af_array &c_f,
                                 const af_array &r_f) {
    const Array<accT> colFilter = castArray<accT>(c_f);
    const Array<accT> rowFilter = castArray<accT>(r_f);
    const Array<accT> signal    = castArray<accT>(s);

    if (colFilter.isScalar() && rowFilter.isScalar()) {
        Array<accT> colArray = detail::tile(colFilter, signal.dims());
        Array<accT> rowArray = detail::tile(rowFilter, signal.dims());

        Array<accT> filter =
            arithOp<accT, af_mul_t>(colArray, rowArray, signal.dims());

        return getHandle(cast<T, accT>(
            arithOp<accT, af_mul_t>(signal, filter, signal.dims())));
    }

    ARG_ASSERT(2, colFilter.isVector());
    ARG_ASSERT(3, rowFilter.isVector());

    return getHandle(
        convolve2<T, accT, expand>(getArray<T>(s), colFilter, rowFilter));
}

template<dim_t baseDim>
AF_BATCH_KIND identifyBatchKind(const dim4 &sDims, const dim4 &fDims) {
    dim_t sn = sDims.ndims();
    dim_t fn = fDims.ndims();

    if (sn == baseDim && fn == baseDim) { return AF_BATCH_NONE; }
    if (sn == baseDim && (fn > baseDim && fn <= AF_MAX_DIMS)) {
        return AF_BATCH_RHS;
    }
    if ((sn > baseDim && sn <= AF_MAX_DIMS) && fn == baseDim) {
        return AF_BATCH_LHS;
    }
    if ((sn > baseDim && sn <= AF_MAX_DIMS) &&
        (fn > baseDim && fn <= AF_MAX_DIMS)) {
        bool doesDimensionsMatch = true;
        bool isInterleaved       = true;
        for (dim_t i = baseDim; i < AF_MAX_DIMS; i++) {
            doesDimensionsMatch &= (sDims[i] == fDims[i]);
            isInterleaved &=
                (sDims[i] == 1 || fDims[i] == 1 || sDims[i] == fDims[i]);
        }
        if (doesDimensionsMatch) { return AF_BATCH_SAME; }
        return (isInterleaved ? AF_BATCH_DIFF : AF_BATCH_UNSUPPORTED);
    }
    return AF_BATCH_UNSUPPORTED;
}

template<dim_t baseDim, bool expand>
af_err convolve(af_array *out, const af_array signal, const af_array filter) {
    try {
        const ArrayInfo &sInfo = getInfo(signal);
        const ArrayInfo &fInfo = getInfo(filter);

        af_dtype stype = sInfo.getType();

        dim4 sdims = sInfo.dims();
        dim4 fdims = fInfo.dims();

        if (fdims.ndims() == 0 || sdims.ndims() == 0) {
            return af_retain_array(out, signal);
        }

        AF_BATCH_KIND convBT = identifyBatchKind<baseDim>(sdims, fdims);

        ARG_ASSERT(1,
                   (convBT != AF_BATCH_UNSUPPORTED && convBT != AF_BATCH_DIFF));

        af_array output;
        switch (stype) {
            case c32:
                output = convolve<cfloat, cfloat, baseDim, expand>(
                    signal, filter, convBT);
                break;
            case c64:
                output = convolve<cdouble, cdouble, baseDim, expand>(
                    signal, filter, convBT);
                break;
            case f32:
                output = convolve<float, float, baseDim, expand>(signal, filter,
                                                                 convBT);
                break;
            case f64:
                output = convolve<double, double, baseDim, expand>(
                    signal, filter, convBT);
                break;
            case u32:
                output = convolve<uint, float, baseDim, expand>(signal, filter,
                                                                convBT);
                break;
            case s32:
                output = convolve<int, float, baseDim, expand>(signal, filter,
                                                               convBT);
                break;
            case u16:
                output = convolve<ushort, float, baseDim, expand>(
                    signal, filter, convBT);
                break;
            case s16:
                output = convolve<short, float, baseDim, expand>(signal, filter,
                                                                 convBT);
                break;
            case u64:
                output = convolve<uintl, float, baseDim, expand>(signal, filter,
                                                                 convBT);
                break;
            case s64:
                output = convolve<intl, float, baseDim, expand>(signal, filter,
                                                                convBT);
                break;
            case u8:
                output = convolve<uchar, float, baseDim, expand>(signal, filter,
                                                                 convBT);
                break;
            case b8:
                output = convolve<char, float, baseDim, expand>(signal, filter,
                                                                convBT);
                break;
            default: TYPE_ERROR(1, stype);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<bool expand>
af_err convolve2_sep(af_array *out, af_array col_filter, af_array row_filter,
                     const af_array signal) {
    try {
        const ArrayInfo &sInfo = getInfo(signal);

        const dim4 &sdims = sInfo.dims();

        const af_dtype signalType = sInfo.getType();

        ARG_ASSERT(1, (sdims.ndims() >= 2));

        af_array output = 0;

        switch (signalType) {
            case c32:
                output = convolve2<cfloat, cfloat, expand>(signal, col_filter,
                                                           row_filter);
                break;
            case c64:
                output = convolve2<cdouble, cdouble, expand>(signal, col_filter,
                                                             row_filter);
                break;
            case f32:
                output = convolve2<float, float, expand>(signal, col_filter,
                                                         row_filter);
                break;
            case f64:
                output = convolve2<double, double, expand>(signal, col_filter,
                                                           row_filter);
                break;
            case u32:
                output = convolve2<uint, float, expand>(signal, col_filter,
                                                        row_filter);
                break;
            case s32:
                output = convolve2<int, float, expand>(signal, col_filter,
                                                       row_filter);
                break;
            case u16:
                output = convolve2<ushort, float, expand>(signal, col_filter,
                                                          row_filter);
                break;
            case s16:
                output = convolve2<short, float, expand>(signal, col_filter,
                                                         row_filter);
                break;
            case u64:
                output = convolve2<uintl, float, expand>(signal, col_filter,
                                                         row_filter);
                break;
            case s64:
                output = convolve2<intl, float, expand>(signal, col_filter,
                                                        row_filter);
                break;
            case u8:
                output = convolve2<uchar, float, expand>(signal, col_filter,
                                                         row_filter);
                break;
            case b8:
                output = convolve2<char, float, expand>(signal, col_filter,
                                                        row_filter);
                break;
            default: TYPE_ERROR(1, signalType);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<int baseDim>
bool isFreqDomain(const af_array &signal, const af_array filter,
                  af_conv_domain domain) {
    if (domain == AF_CONV_FREQ) { return true; }
    if (domain != AF_CONV_AUTO) { return false; }

    const ArrayInfo &sInfo = getInfo(signal);
    const ArrayInfo &fInfo = getInfo(filter);

    const dim4 &sdims = sInfo.dims();
    dim4 fdims        = fInfo.dims();

    if (identifyBatchKind<baseDim>(sdims, fdims) == AF_BATCH_DIFF) {
        return true;
    }

    int kbatch = 1;
    for (int i = 3; i >= baseDim; i--) { kbatch *= fdims[i]; }

    if (kbatch >= 10) { return true; }

    if (baseDim == 1) {
        if (fdims[0] > 128) { return true; }
    }

    if (baseDim == 2) {
        // maximum supported size in 2D domain
        if (fdims[0] > 17 || fdims[1] > 17) { return true; }

        // Maximum supported non square size
        if (fdims[0] != fdims[1] && fdims[0] > 5) { return true; }
    }

    if (baseDim == 3) {
        if (fdims[0] > 5 || fdims[1] > 5 || fdims[2] > 5) { return true; }
    }

    return false;
}

af_err af_convolve1(af_array *out, const af_array signal, const af_array filter,
                    const af_conv_mode mode, af_conv_domain domain) {
    try {
        if (isFreqDomain<1>(signal, filter, domain)) {
            return af_fft_convolve1(out, signal, filter, mode);
        }

        if (mode == AF_CONV_EXPAND) {
            return convolve<1, true>(out, signal, filter);
        }
        { return convolve<1, false>(out, signal, filter); }
    }
    CATCHALL;
}

af_err af_convolve2(af_array *out, const af_array signal, const af_array filter,
                    const af_conv_mode mode, af_conv_domain domain) {
    try {
        if (getInfo(signal).dims().ndims() < 2 ||
            getInfo(filter).dims().ndims() < 2) {
            return af_convolve1(out, signal, filter, mode, domain);
        }

        if (isFreqDomain<2>(signal, filter, domain)) {
            return af_fft_convolve2(out, signal, filter, mode);
        }

        if (mode == AF_CONV_EXPAND) {
            return convolve<2, true>(out, signal, filter);
        } else {
            return convolve<2, false>(out, signal, filter);
        }
    }
    CATCHALL;
}

template<typename T>
inline static af_array convolve2Strided(const af_array &s, const af_array &f,
                                        const dim4 stride, const dim4 padding,
                                        const dim4 dilation) {
    return getHandle(convolve2<T>(getArray<T>(s), getArray<T>(f), stride,
                                  padding, dilation));
}

af_err af_convolve2_nn(af_array *out, const af_array signal,
                       const af_array filter, const unsigned stride_dims,
                       const dim_t *strides, const unsigned padding_dims,
                       const dim_t *paddings, const unsigned dilation_dims,
                       const dim_t *dilations) {
    try {
        const ArrayInfo &sInfo = getInfo(signal);
        const ArrayInfo &fInfo = getInfo(filter);

        af::dim4 sDims = sInfo.dims();
        af::dim4 fDims = fInfo.dims();

        const af_dtype signalType = sInfo.getType();

        ARG_ASSERT(3, stride_dims > 0 && stride_dims <= 2);
        ARG_ASSERT(5, padding_dims > 0 && padding_dims <= 2);
        ARG_ASSERT(7, dilation_dims > 0 && dilation_dims <= 2);

        dim4 stride(stride_dims, strides);
        dim4 padding(padding_dims, paddings);
        dim4 dilation(dilation_dims, dilations);

        // assert number of features matches between signal and filter
        DIM_ASSERT(1, sDims[2] == fDims[2]);

        af_array output;
        switch (signalType) {
            case f32:
                output = convolve2Strided<float>(signal, filter, stride,
                                                 padding, dilation);
                break;
            case f64:
                output = convolve2Strided<double>(signal, filter, stride,
                                                  padding, dilation);
                break;
            case f16:
                output = convolve2Strided<half>(signal, filter, stride, padding,
                                                dilation);
                break;
            default: TYPE_ERROR(1, signalType);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_convolve3(af_array *out, const af_array signal, const af_array filter,
                    const af_conv_mode mode, af_conv_domain domain) {
    try {
        if (getInfo(signal).dims().ndims() < 3 ||
            getInfo(filter).dims().ndims() < 3) {
            return af_convolve2(out, signal, filter, mode, domain);
        }

        if (isFreqDomain<3>(signal, filter, domain)) {
            return af_fft_convolve3(out, signal, filter, mode);
        }

        if (mode == AF_CONV_EXPAND) {
            return convolve<3, true>(out, signal, filter);
        } else {
            return convolve<3, false>(out, signal, filter);
        }
    }
    CATCHALL;
}

af_err af_convolve2_sep(af_array *out, const af_array signal,
                        const af_array col_filter, const af_array row_filter,
                        const af_conv_mode mode) {
    try {
        if (mode == AF_CONV_EXPAND) {
            return convolve2_sep<true>(out, signal, col_filter, row_filter);
        } else {
            return convolve2_sep<false>(out, signal, col_filter, row_filter);
        }
    }
    CATCHALL;
}

template<typename T>
af_array conv2GradCall(const af_array incoming_gradient,
                       const af_array original_signal,
                       const af_array original_filter,
                       const af_array convolved_output, const dim4 &stride,
                       const dim4 &padding, const dim4 &dilation,
                       af_conv_gradient_type grad_type) {
    if (grad_type == AF_CONV_GRADIENT_FILTER) {
        return getHandle(detail::conv2FilterGradient<T>(
            getArray<T>(incoming_gradient), getArray<T>(original_signal),
            getArray<T>(original_filter), getArray<T>(convolved_output), stride,
            padding, dilation));
    } else {
        return getHandle(detail::conv2DataGradient<T>(
            getArray<T>(incoming_gradient), getArray<T>(original_signal),
            getArray<T>(original_filter), getArray<T>(convolved_output), stride,
            padding, dilation));
    }
}

af_err af_convolve2_gradient_nn(
    af_array *out, const af_array incoming_gradient,
    const af_array original_signal, const af_array original_filter,
    const af_array convolved_output, const unsigned stride_dims,
    const dim_t *strides, const unsigned padding_dims, const dim_t *paddings,
    const unsigned dilation_dims, const dim_t *dilations,
    af_conv_gradient_type grad_type) {
    try {
        const ArrayInfo &iinfo = getInfo(incoming_gradient);
        const af::dim4 &iDims  = iinfo.dims();

        const ArrayInfo &sinfo = getInfo(original_signal);
        af::dim4 sDims         = sinfo.dims();

        const ArrayInfo &finfo = getInfo(original_filter);
        af::dim4 fDims         = finfo.dims();

        const ArrayInfo &oinfo = getInfo(convolved_output);
        af::dim4 oDims         = oinfo.dims();

        DIM_ASSERT(1, iDims == oDims);
        DIM_ASSERT(3, oDims[2] == fDims[3]);
        DIM_ASSERT(3, oDims[3] == sDims[3]);
        DIM_ASSERT(2, sDims[2] == fDims[2]);

        af_array output;

        ARG_ASSERT(3, stride_dims > 0 && stride_dims <= 2);
        ARG_ASSERT(5, padding_dims > 0 && padding_dims <= 2);
        ARG_ASSERT(7, dilation_dims > 0 && dilation_dims <= 2);

        af::dim4 stride(stride_dims, strides);
        af::dim4 padding(padding_dims, paddings);
        af::dim4 dilation(dilation_dims, dilations);

        af_dtype type = oinfo.getType();
        switch (type) {
            case f32:
                output = conv2GradCall<float>(
                    incoming_gradient, original_signal, original_filter,
                    convolved_output, stride, padding, dilation, grad_type);
                break;
            case f64:
                output = conv2GradCall<double>(
                    incoming_gradient, original_signal, original_filter,
                    convolved_output, stride, padding, dilation, grad_type);
                break;
            case f16:
                output = conv2GradCall<half>(
                    incoming_gradient, original_signal, original_filter,
                    convolved_output, stride, padding, dilation, grad_type);
                break;
            default: TYPE_ERROR(1, type);
        }
        // output array is pooled array
        std::swap(output, *out);
    }
    CATCHALL;

    return AF_SUCCESS;
}
