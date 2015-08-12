/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/image.h>
#include <af/index.h>
#include <handle.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <arith.hpp>
#include <join.hpp>
#include <math.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static Array<T> mix(const Array<T>& X, const Array<T>& Y,
                double xf, double yf)
{
    dim4 dims = X.dims();
    Array<T> xf_cnst = createValueArray<T>(dims, xf);
    Array<T> yf_cnst = createValueArray<T>(dims, yf);

    Array<T> fX = arithOp<T, af_mul_t>(xf_cnst, X, dims);
    Array<T> fY = arithOp<T, af_mul_t>(yf_cnst, Y, dims);

    return arithOp<T, af_add_t>(fX, fY, dims);
}

template<typename T>
static Array<T> mix(const Array<T>& X, const Array<T>& Y, const Array<T>& Z,
                double xf, double yf, double zf)
{
    dim4 dims = X.dims();
    Array<T> xf_cnst = createValueArray<T>(dims, xf);
    Array<T> yf_cnst = createValueArray<T>(dims, yf);
    Array<T> zf_cnst = createValueArray<T>(dims, zf);

    Array<T> fX = arithOp<T, af_mul_t>(xf_cnst, X, dims);
    Array<T> fY = arithOp<T, af_mul_t>(yf_cnst, Y, dims);
    Array<T> fZ = arithOp<T, af_mul_t>(zf_cnst, Z, dims);

    Array<T> fx_fy = arithOp<T, af_add_t>(fX, fY, dims);
    return arithOp<T, af_add_t>(fx_fy, fZ, dims);
}

template<typename T>
static Array<T> digitize(const Array<T> ch, const double scale, const double offset)
{
    dim4 dims = ch.dims();
    Array<T> base = createValueArray<T>(dims, scalar<T>(offset));
    Array<T> cnst = createValueArray<T>(dims, scalar<T>(scale));
    Array<T> scl  = arithOp<T, af_mul_t>(ch, cnst, dims);
    return arithOp<T, af_add_t>(scl, base, dims);
}

template<typename T, bool isYCbCr2RGB>
static af_array convert(const af_array& in, const af_ycc_std standard)
{
    static const float INV_219 = 0.004566210;
    static const float INV_112 = 0.008928571;
    const static float k[6] = {
        0.1140f, 0.2990f,
        0.0722f, 0.2126f,
        0.0593f, 0.2627f
    };
    unsigned stdIdx = 0; // Default standard is AF_YCC_601
    switch(standard) {
        case AF_YCC_709 : stdIdx = 2; break;
        case AF_YCC_2020: stdIdx = 4; break;
        default         : stdIdx = 0; break;
    }
    float kb = k[stdIdx];
    float kr = k[stdIdx+1];
    float kl = 1.0f - kb - kr;
    float invKl = 1/kl;

    // extract three channels as three slices
    // prepare sequence objects
    af_seq slice1[4] = { af_span, af_span, {0, 0, 1}, af_span };
    af_seq slice2[4] = { af_span, af_span, {1, 1, 1}, af_span };
    af_seq slice3[4] = { af_span, af_span, {2, 2, 1}, af_span };
    // index the array for channels
    af_array ch1Temp=0, ch2Temp=0, ch3Temp=0;
    AF_CHECK(af_index(&ch1Temp, in, 4, slice1));
    AF_CHECK(af_index(&ch2Temp, in, 4, slice2));
    AF_CHECK(af_index(&ch3Temp, in, 4, slice3));
    // get Array objects for corresponding channel views
    Array<T> X = getArray<T>(ch1Temp);
    Array<T> Y = getArray<T>(ch2Temp);
    Array<T> Z = getArray<T>(ch3Temp);

    if (isYCbCr2RGB) {
        dim4 dims = X.dims();
        Array<T> yc  = createValueArray<T>(dims, 16);
        Array<T> cc  = createValueArray<T>(dims, 128);
        Array<T> Y_  = arithOp<T, af_sub_t>(X, yc, dims);
        Array<T> Cb_ = arithOp<T, af_sub_t>(Y, cc, dims);
        Array<T> Cr_ = arithOp<T, af_sub_t>(Z, cc, dims);
        Array<T> R   = mix<T>(Y_, Cr_, INV_219, INV_112*(1-kr));
        Array<T> G   = mix<T>(Y_, Cr_, Cb_,
                              INV_219,
                              INV_112*(kr-1)*kr*invKl,
                              INV_112*(kb-1)*kb*invKl);
        Array<T> B   = mix<T>(Y_, Cb_, INV_219, INV_112*(1-kb));
        // join channels
        Array<T> RG = join<T, T>(2, R, G);
        return getHandle(join<T, T>(2, RG, B));
    } else {
        Array<T> Ey  = mix<T>(X, Y, Z, kr, kl, kb);
        Array<T> Ecr = mix<T>(X, Y, Z, 0.5, 0.5*kl/(kr-1), 0.5*kb/(kr-1));
        Array<T> Ecb = mix<T>(X, Y, Z, 0.5*kr/(kb-1), 0.5*kl/(kb-1), 0.5);
        Array<T> Y = digitize<T>(Ey, 219.0, 16.0);
        Array<T> Cr = digitize<T>(Ecr, 224.0, 128.0);
        Array<T> Cb = digitize<T>(Ecb, 224.0, 128.0);
        // join channels
        Array<T> YCb = join<T, T>(2, Y, Cb);
        return getHandle(join<T, T>(2, YCb, Cr));
    }
}

template<bool isYCbCr2RGB>
af_err convert(af_array* out, const af_array& in, const af_ycc_std standard)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype iType = info.getType();
        af::dim4 inputDims = info.dims();

        ARG_ASSERT(1, (inputDims.ndims() >= 3));

        af_array output = 0;
        switch (iType) {
            case f64: output = convert<double, isYCbCr2RGB>(in, standard); break;
            case f32: output = convert<float , isYCbCr2RGB>(in, standard); break;
            default: TYPE_ERROR(1, iType); break;
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_ycbcr2rgb(af_array* out, const af_array in, const af_ycc_std standard)
{
    return convert<true>(out, in, standard);
}

af_err af_rgb2ycbcr(af_array* out, const af_array in, const af_ycc_std standard)
{
    return convert<false>(out, in, standard);
}
