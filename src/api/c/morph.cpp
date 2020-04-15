/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arith.hpp>
#include <backend.hpp>
#include <cast.hpp>
#include <common/err_common.hpp>
#include <common/indexing_helpers.hpp>
#include <copy.hpp>
#include <fftconvolve.hpp>
#include <handle.hpp>
#include <logic.hpp>
#include <math.hpp>
#include <morph.hpp>
#include <unary.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/image.h>

using af::dim4;
using common::flip;
using detail::arithOp;
using detail::Array;
using detail::cast;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using detail::createValueArray;
using detail::logicOp;
using detail::scalar;
using detail::uchar;
using detail::uint;
using detail::unaryOp;
using detail::ushort;

template<typename T, bool isDilation>
static inline af_array morph(const af_array &in, const af_array &mask) {
    const Array<T> input   = getArray<T>(in);
    const Array<T> &filter = castArray<T>(mask);
    Array<T> out           = morph<T, isDilation>(input, filter);
    return getHandle(out);
}

template<bool isDilation>
static inline af_array morph(const af_array &input, const af_array &mask) {
    using detail::fftconvolve;

#if defined(AF_CPU)
#if defined(USE_MKL)
    constexpr unsigned fftMethodThreshold = 11;
#else
    constexpr unsigned fftMethodThreshold = 27;
#endif  // defined(USE_MKL)
#elif defined(AF_CUDA)
    constexpr unsigned fftMethodThreshold = 17;
#elif defined(AF_OPENCL)
    constexpr unsigned fftMethodThreshold = 19;
#endif  // defined(AF_CPU)

    const Array<float> se = castArray<float>(mask);
    const dim4 &seDims    = se.dims();

    if (seDims[0] <= fftMethodThreshold) {
        return morph<char, isDilation>(input, mask);
    }

    DIM_ASSERT(2, (seDims[0] == seDims[1]));

    const Array<char> in = getArray<char>(input);
    const dim4 &inDims   = in.dims();
    const auto paddedSe =
        padArrayBorders(se,
                        {static_cast<dim_t>(seDims[0] % 2 == 0),
                         static_cast<dim_t>(seDims[1] % 2 == 0), 0, 0},
                        {0, 0, 0, 0}, AF_PAD_ZERO);

    auto fftConv = fftconvolve<float, float, cfloat, false, false, 2>;

    if (isDilation) {
        Array<float> dft =
            fftConv(cast<float>(in), paddedSe, false, AF_BATCH_LHS);

        return getHandle(cast<char>(unaryOp<float, af_round_t>(dft)));
    } else {
        const Array<char> ONES   = createValueArray(inDims, scalar<char>(1));
        const Array<float> ZEROS = createValueArray(inDims, scalar<float>(0));
        const Array<char> inv    = arithOp<char, af_sub_t>(ONES, in, inDims);

        Array<float> dft =
            fftConv(cast<float>(inv), paddedSe, false, AF_BATCH_LHS);

        Array<float> rounded = unaryOp<float, af_round_t>(dft);
        Array<char> thrshd   = logicOp<float, af_gt_t>(rounded, ZEROS, inDims);
        Array<char> inverted = arithOp<char, af_sub_t>(ONES, thrshd, inDims);

        return getHandle(inverted);
    }
}

template<typename T, bool isDilation>
static inline af_array morph3d(const af_array &in, const af_array &mask) {
    const Array<T> input   = getArray<T>(in);
    const Array<T> &filter = castArray<T>(mask);
    Array<T> out           = morph3d<T, isDilation>(input, filter);
    return getHandle(out);
}

template<bool isDilation>
static af_err morph(af_array *out, const af_array &in, const af_array &mask) {
    try {
        const ArrayInfo &info  = getInfo(in);
        const ArrayInfo &mInfo = getInfo(mask);
        af::dim4 dims          = info.dims();
        af::dim4 mdims         = mInfo.dims();
        dim_t in_ndims         = dims.ndims();
        dim_t mask_ndims       = mdims.ndims();

        DIM_ASSERT(1, (in_ndims >= 2));
        DIM_ASSERT(2, (mask_ndims == 2));

        af_array output;
        af_dtype type = info.getType();
        switch (type) {
            case f32: output = morph<float, isDilation>(in, mask); break;
            case f64: output = morph<double, isDilation>(in, mask); break;
            case b8: output = morph<isDilation>(in, mask); break;
            case s32: output = morph<int, isDilation>(in, mask); break;
            case u32: output = morph<uint, isDilation>(in, mask); break;
            case s16: output = morph<short, isDilation>(in, mask); break;
            case u16: output = morph<ushort, isDilation>(in, mask); break;
            case u8: output = morph<uchar, isDilation>(in, mask); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<bool isDilation>
static af_err morph3d(af_array *out, const af_array &in, const af_array &mask) {
    try {
        const ArrayInfo &info  = getInfo(in);
        const ArrayInfo &mInfo = getInfo(mask);
        af::dim4 dims          = info.dims();
        af::dim4 mdims         = mInfo.dims();
        dim_t in_ndims         = dims.ndims();
        dim_t mask_ndims       = mdims.ndims();

        DIM_ASSERT(1, (in_ndims >= 3));
        DIM_ASSERT(2, (mask_ndims == 3));

        af_array output;
        af_dtype type = info.getType();
        switch (type) {
            case f32: output = morph3d<float, isDilation>(in, mask); break;
            case f64: output = morph3d<double, isDilation>(in, mask); break;
            case b8: output = morph3d<char, isDilation>(in, mask); break;
            case s32: output = morph3d<int, isDilation>(in, mask); break;
            case u32: output = morph3d<uint, isDilation>(in, mask); break;
            case s16: output = morph3d<short, isDilation>(in, mask); break;
            case u16: output = morph3d<ushort, isDilation>(in, mask); break;
            case u8: output = morph3d<uchar, isDilation>(in, mask); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
af_err af_dilate(af_array *out, const af_array in, const af_array mask) {
    return morph<true>(out, in, mask);
}

af_err af_erode(af_array *out, const af_array in, const af_array mask) {
    return morph<false>(out, in, mask);
}

af_err af_dilate3(af_array *out, const af_array in, const af_array mask) {
    return morph3d<true>(out, in, mask);
}

af_err af_erode3(af_array *out, const af_array in, const af_array mask) {
    return morph3d<false>(out, in, mask);
}
