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
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <common/tile.hpp>
#include <fftconvolve.hpp>
#include <handle.hpp>
#include <af/data.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/ml.h>
#include <af/signal.h>

#include <cstdio>

using af::dim4;
using arrayfire::common::cast;
using arrayfire::common::half;
using detail::arithOp;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::convolve;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T, typename accT>
inline af_array convolve(const af_array &s, const af_array &f,
                         AF_BATCH_KIND kind, const int rank,
                         const bool expand) {
    return getHandle(convolve<T, accT>(getArray<T>(s), castArray<accT>(f), kind,
                                       rank, expand));
}

template<typename T, typename accT>
inline af_array convolve2(const af_array &s, const af_array &c_f,
                          const af_array &r_f, const bool expand) {
    const Array<accT> colFilter = castArray<accT>(c_f);
    const Array<accT> rowFilter = castArray<accT>(r_f);
    const Array<accT> signal    = castArray<accT>(s);

    if (colFilter.isScalar() && rowFilter.isScalar()) {
        Array<accT> colArray =
            arrayfire::common::tile(colFilter, signal.dims());
        Array<accT> rowArray =
            arrayfire::common::tile(rowFilter, signal.dims());

        Array<accT> filter =
            arithOp<accT, af_mul_t>(colArray, rowArray, signal.dims());

        return getHandle(cast<T, accT>(
            arithOp<accT, af_mul_t>(signal, filter, signal.dims())));
    }

    ARG_ASSERT(2, colFilter.isVector());
    ARG_ASSERT(3, rowFilter.isVector());

    return getHandle(
        convolve2<T, accT>(getArray<T>(s), colFilter, rowFilter, expand));
}

AF_BATCH_KIND identifyBatchKind(const int rank, const dim4 &sDims,
                                const dim4 &fDims) {
    dim_t sn = sDims.ndims();
    dim_t fn = fDims.ndims();

    if (sn == rank && fn == rank) { return AF_BATCH_NONE; }
    if (sn == rank && (fn > rank && fn <= AF_MAX_DIMS)) { return AF_BATCH_RHS; }
    if ((sn > rank && sn <= AF_MAX_DIMS) && fn == rank) { return AF_BATCH_LHS; }
    if ((sn > rank && sn <= AF_MAX_DIMS) && (fn > rank && fn <= AF_MAX_DIMS)) {
        bool doesDimensionsMatch = true;
        bool isInterleaved       = true;
        for (dim_t i = rank; i < AF_MAX_DIMS; i++) {
            doesDimensionsMatch &= (sDims[i] == fDims[i]);
            isInterleaved &=
                (sDims[i] == 1 || fDims[i] == 1 || sDims[i] == fDims[i]);
        }
        if (doesDimensionsMatch) { return AF_BATCH_SAME; }
        return (isInterleaved ? AF_BATCH_DIFF : AF_BATCH_UNSUPPORTED);
    }
    return AF_BATCH_UNSUPPORTED;
}

bool isFreqDomain(const int rank, const af_array &signal, const af_array filter,
                  af_conv_domain domain) {
    if (domain == AF_CONV_FREQ) { return true; }
    if (domain != AF_CONV_AUTO) { return false; }

    const ArrayInfo &sInfo = getInfo(signal);
    const ArrayInfo &fInfo = getInfo(filter);

    const dim4 &sdims = sInfo.dims();
    dim4 fdims        = fInfo.dims();

    if (identifyBatchKind(rank, sdims, fdims) == AF_BATCH_DIFF) { return true; }

    int kbatch = 1;
    for (int i = 3; i >= rank; i--) { kbatch *= fdims[i]; }

    if (kbatch >= 10) { return true; }
    if (rank == 1) {
        if (fdims[0] > 128) { return true; }
    }
    if (rank == 2) {
        // maximum supported size in 2D domain
        if (fdims[0] > 17 || fdims[1] > 17) { return true; }

        // Maximum supported non square size
        if (fdims[0] != fdims[1] && fdims[0] > 5) { return true; }
    }
    if (rank == 3) {
        if (fdims[0] > 5 || fdims[1] > 5 || fdims[2] > 5) { return true; }
    }
    return false;
}

af_err convolve(af_array *out, const af_array signal, const af_array filter,
                const af_conv_mode mode, const int rank) {
    try {
        const ArrayInfo &sInfo = getInfo(signal);
        const ArrayInfo &fInfo = getInfo(filter);

        af_dtype stype = sInfo.getType();

        dim4 sdims = sInfo.dims();
        dim4 fdims = fInfo.dims();

        if (fdims.ndims() == 0 || sdims.ndims() == 0) {
            return af_retain_array(out, signal);
        }

        AF_BATCH_KIND convBT = identifyBatchKind(rank, sdims, fdims);

        ARG_ASSERT(1,
                   (convBT != AF_BATCH_UNSUPPORTED && convBT != AF_BATCH_DIFF));

        const bool expand = mode == AF_CONV_EXPAND;

        af_array output;
        switch (stype) {
            case c32:
                output = convolve<cfloat, cfloat>(signal, filter, convBT, rank,
                                                  expand);
                break;
            case c64:
                output = convolve<cdouble, cdouble>(signal, filter, convBT,
                                                    rank, expand);
                break;
            case f32:
                output = convolve<float, float>(signal, filter, convBT, rank,
                                                expand);
                break;
            case f64:
                output = convolve<double, double>(signal, filter, convBT, rank,
                                                  expand);
                break;
            case u32:
                output =
                    convolve<uint, float>(signal, filter, convBT, rank, expand);
                break;
            case s32:
                output =
                    convolve<int, float>(signal, filter, convBT, rank, expand);
                break;
            case u16:
                output = convolve<ushort, float>(signal, filter, convBT, rank,
                                                 expand);
                break;
            case s16:
                output = convolve<short, float>(signal, filter, convBT, rank,
                                                expand);
                break;
            case u64:
                output = convolve<uintl, float>(signal, filter, convBT, rank,
                                                expand);
                break;
            case s64:
                output =
                    convolve<intl, float>(signal, filter, convBT, rank, expand);
                break;
            case u8:
                output = convolve<uchar, float>(signal, filter, convBT, rank,
                                                expand);
                break;
            case b8:
                output =
                    convolve<char, float>(signal, filter, convBT, rank, expand);
                break;
            default: TYPE_ERROR(1, stype);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_convolve1(af_array *out, const af_array signal, const af_array filter,
                    const af_conv_mode mode, af_conv_domain domain) {
    try {
        if (isFreqDomain(1, signal, filter, domain)) {
            return af_fft_convolve1(out, signal, filter, mode);
        }
        return convolve(out, signal, filter, mode, 1);
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
        if (isFreqDomain(2, signal, filter, domain)) {
            return af_fft_convolve2(out, signal, filter, mode);
        }
        return convolve(out, signal, filter, mode, 2);
    }
    CATCHALL;
}

af_err af_convolve3(af_array *out, const af_array signal, const af_array filter,
                    const af_conv_mode mode, af_conv_domain domain) {
    try {
        if (getInfo(signal).dims().ndims() < 3 ||
            getInfo(filter).dims().ndims() < 3) {
            return af_convolve2(out, signal, filter, mode, domain);
        }
        if (isFreqDomain(3, signal, filter, domain)) {
            return af_fft_convolve3(out, signal, filter, mode);
        }
        return convolve(out, signal, filter, mode, 3);
    }
    CATCHALL;
}

af_err af_convolve2_sep(af_array *out, const af_array col_filter,
                        const af_array row_filter, const af_array signal,
                        const af_conv_mode mode) {
    try {
        const ArrayInfo &sInfo = getInfo(signal);

        const dim4 &sdims = sInfo.dims();

        const af_dtype signalType = sInfo.getType();

        ARG_ASSERT(1, (sdims.ndims() >= 2));

        af_array output = 0;

        const bool expand = mode == AF_CONV_EXPAND;

        switch (signalType) {
            case c32:
                output = convolve2<cfloat, cfloat>(signal, col_filter,
                                                   row_filter, expand);
                break;
            case c64:
                output = convolve2<cdouble, cdouble>(signal, col_filter,
                                                     row_filter, expand);
                break;
            case f32:
                output = convolve2<float, float>(signal, col_filter, row_filter,
                                                 expand);
                break;
            case f64:
                output = convolve2<double, double>(signal, col_filter,
                                                   row_filter, expand);
                break;
            case u32:
                output = convolve2<uint, float>(signal, col_filter, row_filter,
                                                expand);
                break;
            case s32:
                output = convolve2<int, float>(signal, col_filter, row_filter,
                                               expand);
                break;
            case u16:
                output = convolve2<ushort, float>(signal, col_filter,
                                                  row_filter, expand);
                break;
            case s16:
                output = convolve2<short, float>(signal, col_filter, row_filter,
                                                 expand);
                break;
            case u64:
                output = convolve2<uintl, float>(signal, col_filter, row_filter,
                                                 expand);
                break;
            case s64:
                output = convolve2<intl, float>(signal, col_filter, row_filter,
                                                expand);
                break;
            case u8:
                output = convolve2<uchar, float>(signal, col_filter, row_filter,
                                                 expand);
                break;
            case b8:
                output = convolve2<char, float>(signal, col_filter, row_filter,
                                                expand);
                break;
            default: TYPE_ERROR(1, signalType);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
inline af_array convolve2Strided(const af_array &s, const af_array &f,
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

        dim4 stride(stride_dims, strides);
        dim4 padding(padding_dims, paddings);
        dim4 dilation(dilation_dims, dilations);

        size_t stride_ndims   = stride.ndims();
        size_t padding_ndims  = padding.ndims();
        size_t dilation_ndims = dilation.ndims();
        ARG_ASSERT(3, stride_ndims > 0 && stride_ndims <= 2);
        ARG_ASSERT(5, padding_ndims >= 0 && padding_ndims <= 2);
        ARG_ASSERT(7, dilation_ndims > 0 && dilation_ndims <= 2);

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

        af::dim4 stride(stride_dims, strides);
        af::dim4 padding(padding_dims, paddings);
        af::dim4 dilation(dilation_dims, dilations);

        size_t stride_ndims   = stride.ndims();
        size_t padding_ndims  = padding.ndims();
        size_t dilation_ndims = dilation.ndims();
        ARG_ASSERT(3, stride_ndims > 0 && stride_ndims <= 2);
        ARG_ASSERT(5, padding_ndims > 0 && padding_ndims <= 2);
        ARG_ASSERT(7, dilation_ndims > 0 && dilation_ndims <= 2);

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
