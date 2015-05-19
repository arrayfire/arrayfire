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
#include <handle.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <fft.hpp>

using af::dim4;
using namespace detail;

template<typename inType, typename outType, int rank, bool isR2C>
static af_array fft(const af_array in, const double norm_factor, const dim_t npad, const dim_t  * const pad)
{
    return getHandle(fft<inType, outType, rank, isR2C>(getArray<inType>(in), norm_factor, npad, pad));
}

template<typename T, int rank>
static af_array ifft(const af_array in, const double norm_factor, const dim_t npad, const dim_t * const pad)
{
    return getHandle(ifft<T, rank>(getArray<T>(in), norm_factor, npad, pad));
}

template<int rank>
static af_err fft(af_array *out, const af_array in, const double norm_factor, const dim_t npad, const dim_t * const pad)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type  = info.getType();
        af::dim4 dims  = info.dims();

        DIM_ASSERT(1, (dims.ndims()>=rank));

        af_array output;
        switch(type) {
            case c32: output = fft<cfloat , cfloat , rank, false>(in, norm_factor, npad, pad); break;
            case c64: output = fft<cdouble, cdouble, rank, false>(in, norm_factor, npad, pad); break;
            case f32: output = fft<float , cfloat  , rank, true >(in, norm_factor, npad, pad); break;
            case f64: output = fft<double, cdouble , rank, true >(in, norm_factor, npad, pad); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<int rank>
static af_err ifft(af_array *out, const af_array in, const double norm_factor, const dim_t npad, const dim_t * const pad)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type  = info.getType();
        af::dim4 dims  = info.dims();

        DIM_ASSERT(1, (dims.ndims()>=rank));

        af_array output;
        switch(type) {
            case c32: output = ifft<cfloat , rank>(in, norm_factor, npad, pad); break;
            case c64: output = ifft<cdouble, rank>(in, norm_factor, npad, pad); break;
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
    return fft<1>(out, in, norm_factor, (pad0>0?1:0), pad);
}

af_err af_fft2(af_array *out, const af_array in, const double norm_factor, const dim_t pad0, const dim_t pad1)
{
    const dim_t pad[2] = {pad0, pad1};
    return fft<2>(out, in, norm_factor, (pad0>0&&pad1>0?2:0), pad);
}

af_err af_fft3(af_array *out, const af_array in, const double norm_factor, const dim_t pad0, const dim_t pad1, const dim_t pad2)
{
    const dim_t pad[3] = {pad0, pad1, pad2};
    return fft<3>(out, in, norm_factor, (pad0>0&&pad1>0&&pad2>0?3:0), pad);
}

af_err af_ifft(af_array *out, const af_array in, const double norm_factor, const dim_t pad0)
{
    const dim_t pad[1] = {pad0};
    return ifft<1>(out, in, norm_factor, (pad0>0?1:0), pad);
}

af_err af_ifft2(af_array *out, const af_array in, const double norm_factor, const dim_t pad0, const dim_t pad1)
{
    const dim_t pad[2] = {pad0, pad1};
    return ifft<2>(out, in, norm_factor, (pad0>0&&pad1>0?2:0), pad);
}

af_err af_ifft3(af_array *out, const af_array in, const double norm_factor, const dim_t pad0, const dim_t pad1, const dim_t pad2)
{
    const dim_t pad[3] = {pad0, pad1, pad2};
    return ifft<3>(out, in, norm_factor, (pad0>0&&pad1>0&&pad2>0?3:0), pad);
}
