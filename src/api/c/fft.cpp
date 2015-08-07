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
#include <af/signal.h>
#include <err_common.hpp>
#include <backend.hpp>
#include <fft_common.hpp>

using af::dim4;
using namespace detail;

template<typename inType, typename outType, int rank, bool direction>
static af_array fft(const af_array in, const double norm_factor,
                    const dim_t npad, const dim_t  * const pad)
{
    return getHandle(fft<inType, outType, rank, direction>(getArray<inType>(in),
                                                           norm_factor, npad, pad));
}

template<int rank, bool direction>
static af_err fft(af_array *out, const af_array in, const double norm_factor, const dim_t npad, const dim_t * const pad)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type  = info.getType();
        af::dim4 dims  = info.dims();

        DIM_ASSERT(1, (dims.ndims()>=rank));

        af_array output;
        switch(type) {
            case c32: output = fft<cfloat , cfloat , rank, direction>(in, norm_factor, npad, pad); break;
            case c64: output = fft<cdouble, cdouble, rank, direction>(in, norm_factor, npad, pad); break;
            case f32: output = fft<float , cfloat  , rank, direction>(in, norm_factor, npad, pad); break;
            case f64: output = fft<double, cdouble , rank, direction>(in, norm_factor, npad, pad); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_fft(af_array *out, const af_array in, const double norm_factor, const dim_t pad0)
{
    const dim_t pad[1] = {pad0};
    return fft<1, true>(out, in, norm_factor, (pad0>0?1:0), pad);
}

af_err af_fft2(af_array *out, const af_array in, const double norm_factor, const dim_t pad0, const dim_t pad1)
{
    const dim_t pad[2] = {pad0, pad1};
    return fft<2, true>(out, in, norm_factor, (pad0>0&&pad1>0?2:0), pad);
}

af_err af_fft3(af_array *out, const af_array in, const double norm_factor, const dim_t pad0, const dim_t pad1, const dim_t pad2)
{
    const dim_t pad[3] = {pad0, pad1, pad2};
    return fft<3, true>(out, in, norm_factor, (pad0>0&&pad1>0&&pad2>0?3:0), pad);
}

af_err af_ifft(af_array *out, const af_array in, const double norm_factor, const dim_t pad0)
{
    const dim_t pad[1] = {pad0};
    return fft<1, false>(out, in, norm_factor, (pad0>0?1:0), pad);
}

af_err af_ifft2(af_array *out, const af_array in, const double norm_factor, const dim_t pad0, const dim_t pad1)
{
    const dim_t pad[2] = {pad0, pad1};
    return fft<2, false>(out, in, norm_factor, (pad0>0&&pad1>0?2:0), pad);
}

af_err af_ifft3(af_array *out, const af_array in, const double norm_factor, const dim_t pad0, const dim_t pad1, const dim_t pad2)
{
    const dim_t pad[3] = {pad0, pad1, pad2};
    return fft<3, false>(out, in, norm_factor, (pad0>0&&pad1>0&&pad2>0?3:0), pad);
}

template<typename T, int rank, bool direction>
static void fft_inplace(const af_array in, const double norm_factor)
{
    Array<T> &input = getWritableArray<T>(in);
    fft_inplace<T, rank, direction>(input);
    if (norm_factor != 1) {
        multiply_inplace<T>(input, norm_factor);
    }
}

template<int rank, bool direction>
static af_err fft_inplace(af_array in, const double norm_factor)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type  = info.getType();
        af::dim4 dims  = info.dims();

        DIM_ASSERT(1, (dims.ndims()>=rank));

        switch(type) {
            case c32: fft_inplace<cfloat , rank, direction>(in, norm_factor); break;
            case c64: fft_inplace<cdouble, rank, direction>(in, norm_factor); break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_fft_inplace(af_array in, const double norm_factor)
{
    return fft_inplace<1, true>(in, norm_factor);
}

af_err af_fft2_inplace(af_array in, const double norm_factor)
{
    return fft_inplace<2, true>(in, norm_factor);
}

af_err af_fft3_inplace(af_array in, const double norm_factor)
{
    return fft_inplace<3, true>(in, norm_factor);
}

af_err af_ifft_inplace(af_array in, const double norm_factor)
{
    return fft_inplace<1, false>(in, norm_factor);
}

af_err af_ifft2_inplace(af_array in, const double norm_factor)
{
    return fft_inplace<2, false>(in, norm_factor);
}

af_err af_ifft3_inplace(af_array in, const double norm_factor)
{
    return fft_inplace<3, false>(in, norm_factor);
}

template<typename inType, typename outType, int rank>
static af_array fft_r2c(const af_array in, const double norm_factor,
                        const dim_t npad, const dim_t  * const pad)
{
    return getHandle(fft_r2c<inType, outType, rank>(getArray<inType>(in),
                                                    norm_factor, npad, pad));
}

template<int rank>
static af_err fft_r2c(af_array *out, const af_array in, const double norm_factor, const dim_t npad, const dim_t * const pad)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type  = info.getType();
        af::dim4 dims  = info.dims();

        DIM_ASSERT(1, (dims.ndims()>=rank));

        af_array output;
        switch(type) {
            case f32: output = fft_r2c<float , cfloat  , rank>(in, norm_factor, npad, pad); break;
            case f64: output = fft_r2c<double, cdouble , rank>(in, norm_factor, npad, pad); break;
        default: {
            TYPE_ERROR(1, type);
        }
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_fft_r2c(af_array *out, const af_array in, const double norm_factor, const dim_t pad0)
{
    const dim_t pad[1] = {pad0};
    return fft_r2c<1>(out, in, norm_factor, (pad0>0?1:0), pad);
}

af_err af_fft2_r2c(af_array *out, const af_array in, const double norm_factor, const dim_t pad0, const dim_t pad1)
{
    const dim_t pad[2] = {pad0, pad1};
    return fft_r2c<2>(out, in, norm_factor, (pad0>0&&pad1>0?2:0), pad);
}

af_err af_fft3_r2c(af_array *out, const af_array in, const double norm_factor, const dim_t pad0, const dim_t pad1, const dim_t pad2)
{
    const dim_t pad[3] = {pad0, pad1, pad2};
    return fft_r2c<3>(out, in, norm_factor, (pad0>0&&pad1>0&&pad2>0?3:0), pad);
}


template<typename inType, typename outType, int rank>
static af_array fft_c2r(const af_array in, const double norm_factor,
                        const dim4 &odims)
{
    return getHandle(fft_c2r<inType, outType, rank>(getArray<inType>(in),
                                                    norm_factor, odims));
}

template<int rank>
static af_err fft_c2r(af_array *out, const af_array in, const double norm_factor, const bool is_odd)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type  = info.getType();
        af::dim4 idims  = info.dims();

        DIM_ASSERT(1, (idims.ndims()>=rank));

        dim4 odims = idims;
        odims[0] = 2 * (odims[0] - 1) + (is_odd ? 1 : 0);

        af_array output;
        switch(type) {
            case c32: output = fft_c2r<cfloat , float  , rank>(in, norm_factor, odims); break;
            case c64: output = fft_c2r<cdouble, double , rank>(in, norm_factor, odims); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_fft_c2r(af_array *out, const af_array in, const double norm_factor, const bool is_odd)
{
    return fft_c2r<1>(out, in, norm_factor, is_odd);
}

af_err af_fft2_c2r(af_array *out, const af_array in, const double norm_factor, const bool is_odd)
{
    return fft_c2r<2>(out, in, norm_factor, is_odd);
}

af_err af_fft3_c2r(af_array *out, const af_array in, const double norm_factor, const bool is_odd)
{
    return fft_c2r<3>(out, in, norm_factor, is_odd);
}
