/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fftconvolve.hpp>

#include <arith.hpp>
#include <backend.hpp>
#include <common/cast.hpp>
#include <common/dispatch.hpp>
#include <common/err_common.hpp>
#include <complex.hpp>
#include <fft_common.hpp>
#include <handle.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/signal.h>

#include <algorithm>
#include <type_traits>
#include <vector>

using af::dim4;
using arrayfire::common::cast;
using detail::arithOp;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createSubArray;
using detail::fftconvolve;
using detail::intl;
using detail::real;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;
using std::conditional;
using std::is_integral;
using std::is_same;
using std::max;
using std::swap;
using std::vector;

template<typename T>
af_array fftconvolve_fallback(const af_array signal, const af_array filter,
                              const bool expand, const int baseDim) {
    using convT = typename conditional<is_integral<T>::value ||
                                           is_same<T, float>::value ||
                                           is_same<T, cfloat>::value,
                                       float, double>::type;
    using cT    = typename conditional<is_same<convT, float>::value, cfloat,
                                    cdouble>::type;

    const Array<cT> S = castArray<cT>(signal);
    const Array<cT> F = castArray<cT>(filter);
    const dim4 &sdims = S.dims();
    const dim4 &fdims = F.dims();
    dim4 odims(1, 1, 1, 1);
    dim4 psdims(1, 1, 1, 1);
    dim4 pfdims(1, 1, 1, 1);

    vector<af_seq> index(AF_MAX_DIMS);

    int count = 1;
    for (int i = 0; i < baseDim; i++) {
        dim_t tdim_i = sdims[i] + fdims[i] - 1;

        // Pad temporary buffers to power of 2 for performance
        odims[i]  = nextpow2(tdim_i);
        psdims[i] = nextpow2(tdim_i);
        pfdims[i] = nextpow2(tdim_i);

        // The normalization factor
        count *= odims[i];

        // Get the indexing params for output
        if (expand) {
            index[i].begin = 0.;
            index[i].end   = static_cast<double>(tdim_i) - 1.;
        } else {
            index[i].begin = static_cast<double>(fdims[i]) / 2.0;
            index[i].end = static_cast<double>(index[i].begin + sdims[i]) - 1.;
        }
        index[i].step = 1.;
    }

    for (int i = baseDim; i < AF_MAX_DIMS; i++) {
        odims[i]  = max(sdims[i], fdims[i]);
        psdims[i] = sdims[i];
        pfdims[i] = fdims[i];
        index[i]  = af_span;
    }

    // fft(signal)
    Array<cT> T1 = fft<cT, cT>(S, 1.0, baseDim, psdims.get(), baseDim, true);

    // fft(filter)
    Array<cT> T2 = fft<cT, cT>(F, 1.0, baseDim, pfdims.get(), baseDim, true);

    // fft(signal) * fft(filter)
    T1 = arithOp<cT, af_mul_t>(T1, T2, odims);

    // ifft(ffit(signal) * fft(filter))
    T1 = fft<cT, cT>(T1, 1.0 / static_cast<double>(count), baseDim, odims.get(),
                     baseDim, false);

    // Index to proper offsets
    T1 = createSubArray<cT>(T1, index);

    if (getInfo(signal).isComplex() || getInfo(filter).isComplex()) {
        return getHandle(cast<T>(T1));
    } else {
        return getHandle(cast<T>(real<convT>(T1)));
    }
}

template<typename T>
inline af_array fftconvolve(const af_array &s, const af_array &f,
                            const bool expand, AF_BATCH_KIND kind,
                            const int baseDim) {
    if (kind == AF_BATCH_DIFF) {
        return fftconvolve_fallback<T>(s, f, expand, baseDim);
    } else {
        return getHandle(fftconvolve<T>(getArray<T>(s), castArray<T>(f), expand,
                                        kind, baseDim));
    }
}

AF_BATCH_KIND identifyBatchKind(const dim4 &sDims, const dim4 &fDims,
                                const int baseDim) {
    dim_t sn = sDims.ndims();
    dim_t fn = fDims.ndims();

    if (sn == baseDim && fn == baseDim) { return AF_BATCH_NONE; }
    if (sn == baseDim && (fn > baseDim && fn <= AF_MAX_DIMS)) {
        return AF_BATCH_RHS;
    }
    if ((sn > baseDim && sn <= AF_MAX_DIMS) && fn == baseDim) {
        return AF_BATCH_LHS;
    } else if ((sn > baseDim && sn <= AF_MAX_DIMS) &&
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
    } else {
        return AF_BATCH_UNSUPPORTED;
    }
}

af_err fft_convolve(af_array *out, const af_array signal, const af_array filter,
                    const bool expand, const int baseDim) {
    try {
        const ArrayInfo &sInfo = getInfo(signal);
        const ArrayInfo &fInfo = getInfo(filter);

        af_dtype signalType = sInfo.getType();

        const dim4 &sdims = sInfo.dims();
        const dim4 &fdims = fInfo.dims();

        AF_BATCH_KIND convBT = identifyBatchKind(sdims, fdims, baseDim);

        ARG_ASSERT(1, (convBT != AF_BATCH_UNSUPPORTED));

        af_array output;
        switch (signalType) {
            case f64:
                output = fftconvolve<double>(signal, filter, expand, convBT,
                                             baseDim);
                break;
            case f32:
                output =
                    fftconvolve<float>(signal, filter, expand, convBT, baseDim);
                break;
            case u32:
                output =
                    fftconvolve<uint>(signal, filter, expand, convBT, baseDim);
                break;
            case s32:
                output =
                    fftconvolve<int>(signal, filter, expand, convBT, baseDim);
                break;
            case u64:
                output =
                    fftconvolve<uintl>(signal, filter, expand, convBT, baseDim);
                break;
            case s64:
                output =
                    fftconvolve<intl>(signal, filter, expand, convBT, baseDim);
                break;
            case u16:
                output = fftconvolve<ushort>(signal, filter, expand, convBT,
                                             baseDim);
                break;
            case s16:
                output =
                    fftconvolve<short>(signal, filter, expand, convBT, baseDim);
                break;
            case u8:
                output =
                    fftconvolve<uchar>(signal, filter, expand, convBT, baseDim);
                break;
            case b8:
                output =
                    fftconvolve<char>(signal, filter, expand, convBT, baseDim);
                break;
            case c32:
                output = fftconvolve_fallback<cfloat>(signal, filter, expand,
                                                      baseDim);
                break;
            case c64:
                output = fftconvolve_fallback<cdouble>(signal, filter, expand,
                                                       baseDim);
                break;
            default: TYPE_ERROR(1, signalType);
        }
        swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_fft_convolve1(af_array *out, const af_array signal,
                        const af_array filter, const af_conv_mode mode) {
    return fft_convolve(out, signal, filter, mode == AF_CONV_EXPAND, 1);
}

af_err af_fft_convolve2(af_array *out, const af_array signal,
                        const af_array filter, const af_conv_mode mode) {
    if (getInfo(signal).dims().ndims() < 2 &&
        getInfo(filter).dims().ndims() < 2) {
        return fft_convolve(out, signal, filter, mode == AF_CONV_EXPAND, 1);
    }
    return fft_convolve(out, signal, filter, mode == AF_CONV_EXPAND, 2);
}

af_err af_fft_convolve3(af_array *out, const af_array signal,
                        const af_array filter, const af_conv_mode mode) {
    if (getInfo(signal).dims().ndims() < 3 &&
        getInfo(filter).dims().ndims() < 3) {
        return fft_convolve(out, signal, filter, mode == AF_CONV_EXPAND, 2);
    }
    return fft_convolve(out, signal, filter, mode == AF_CONV_EXPAND, 3);
}
