/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/image.h>
#include <handle.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <morph.hpp>

using af::dim4;
using namespace detail;

template<typename T, bool isDilation>
static inline af_array morph(const af_array &in, const af_array &mask)
{
    const Array<T> &input = getArray<T>(in);
    const Array<T> &filter = castArray<T>(mask);
    Array<T> out = morph<T, isDilation>(input, filter);
    return getHandle(out);
}

template<typename T, bool isDilation>
static inline af_array morph3d(const af_array &in, const af_array &mask)
{
    const Array<T> &input = getArray<T>(in);
    const Array<T> &filter = castArray<T>(mask);
    Array<T> out = morph3d<T, isDilation>(input, filter);
    return getHandle(out);
}

template<bool isDilation>
static af_err morph(af_array *out, const af_array &in, const af_array &mask)
{
    try {
        ArrayInfo info = getInfo(in);
        ArrayInfo mInfo= getInfo(mask);
        af::dim4 dims  = info.dims();
        af::dim4 mdims = mInfo.dims();
        dim_t in_ndims = dims.ndims();
        dim_t mask_ndims = mdims.ndims();

        DIM_ASSERT(1, (in_ndims >= 2));
        DIM_ASSERT(2, (mask_ndims == 2));

        af_array output;
        af_dtype type  = info.getType();
        switch(type) {
            case f32: output = morph<float , isDilation>(in, mask);      break;
            case f64: output = morph<double, isDilation>(in, mask);      break;
            case b8 : output = morph<char  , isDilation>(in, mask);      break;
            case s32: output = morph<int   , isDilation>(in, mask);      break;
            case u32: output = morph<uint  , isDilation>(in, mask);      break;
            case u8 : output = morph<uchar , isDilation>(in, mask);      break;
            default : TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<bool isDilation>
static af_err morph3d(af_array *out, const af_array &in, const af_array &mask)
{
    try {
        ArrayInfo info = getInfo(in);
        ArrayInfo mInfo= getInfo(mask);
        af::dim4 dims  = info.dims();
        af::dim4 mdims = mInfo.dims();
        dim_t in_ndims = dims.ndims();
        dim_t mask_ndims = mdims.ndims();

        DIM_ASSERT(1, (in_ndims >= 3));
        DIM_ASSERT(2, (mask_ndims == 3));

        af_array output;
        af_dtype type  = info.getType();
        switch(type) {
            case f32: output = morph3d<float , isDilation>(in, mask);       break;
            case f64: output = morph3d<double, isDilation>(in, mask);       break;
            case b8 : output = morph3d<char  , isDilation>(in, mask);       break;
            case s32: output = morph3d<int   , isDilation>(in, mask);       break;
            case u32: output = morph3d<uint  , isDilation>(in, mask);       break;
            case u8 : output = morph3d<uchar , isDilation>(in, mask);       break;
            default : TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
af_err af_dilate(af_array *out, const af_array in, const af_array mask)
{
    return morph<true>(out,in,mask);
}

af_err af_erode(af_array *out, const af_array in, const af_array mask)
{
    return morph<false>(out,in,mask);
}

af_err af_dilate3(af_array *out, const af_array in, const af_array mask)
{
    return morph3d<true>(out,in,mask);
}

af_err af_erode3(af_array *out, const af_array in, const af_array mask)
{
    return morph3d<false>(out,in,mask);
}
