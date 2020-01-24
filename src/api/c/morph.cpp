/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <morph.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/image.h>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array morph(const af_array &in, const af_array &mask,
                             bool isDilation) {
    const Array<T> &input  = getArray<T>(in);
    const Array<T> &filter = castArray<T>(mask);
    Array<T> out           = morph<T>(input, filter, isDilation);
    return getHandle(out);
}

template<typename T>
static inline af_array morph3d(const af_array &in, const af_array &mask,
                               bool isDilation) {
    const Array<T> &input  = getArray<T>(in);
    const Array<T> &filter = castArray<T>(mask);
    Array<T> out           = morph3d<T>(input, filter, isDilation);
    return getHandle(out);
}

af_err morph(af_array *out, const af_array &in, const af_array &mask,
             bool isDilation) {
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
            case f32: output = morph<float>(in, mask, isDilation); break;
            case f64: output = morph<double>(in, mask, isDilation); break;
            case b8: output = morph<char>(in, mask, isDilation); break;
            case s32: output = morph<int>(in, mask, isDilation); break;
            case u32: output = morph<uint>(in, mask, isDilation); break;
            case s16: output = morph<short>(in, mask, isDilation); break;
            case u16: output = morph<ushort>(in, mask, isDilation); break;
            case u8: output = morph<uchar>(in, mask, isDilation); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err morph3d(af_array *out, const af_array &in, const af_array &mask,
               bool isDilation) {
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
            case f32: output = morph3d<float>(in, mask, isDilation); break;
            case f64: output = morph3d<double>(in, mask, isDilation); break;
            case b8: output = morph3d<char>(in, mask, isDilation); break;
            case s32: output = morph3d<int>(in, mask, isDilation); break;
            case u32: output = morph3d<uint>(in, mask, isDilation); break;
            case s16: output = morph3d<short>(in, mask, isDilation); break;
            case u16: output = morph3d<ushort>(in, mask, isDilation); break;
            case u8: output = morph3d<uchar>(in, mask, isDilation); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_dilate(af_array *out, const af_array in, const af_array mask) {
    return morph(out, in, mask, true);
}

af_err af_erode(af_array *out, const af_array in, const af_array mask) {
    return morph(out, in, mask, false);
}

af_err af_dilate3(af_array *out, const af_array in, const af_array mask) {
    return morph3d(out, in, mask, true);
}

af_err af_erode3(af_array *out, const af_array in, const af_array mask) {
    return morph3d(out, in, mask, false);
}
