/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/signal.h>
#include "symbol_manager.hpp"

af_err af_approx1(af_array *out, const af_array in, const af_array pos, const af_interp_type method, const float offGrid)
{
    CHECK_ARRAYS(in, pos);
    return CALL(out, in, pos, method, offGrid);
}

af_err af_approx2(af_array *out, const af_array in, const af_array pos0, const af_array pos1, const af_interp_type method, const float offGrid)
{
    CHECK_ARRAYS(in, pos0, pos1);
    return CALL(out, in, pos0, pos1, method, offGrid);
}

#define FFT_HAPI_DEF(af_func)\
af_err af_func(af_array in, const double norm_factor)\
{\
    CHECK_ARRAYS(in); \
    return CALL(in, norm_factor);\
}

FFT_HAPI_DEF(af_fft_inplace)
FFT_HAPI_DEF(af_fft2_inplace)
FFT_HAPI_DEF(af_fft3_inplace)
FFT_HAPI_DEF(af_ifft_inplace)
FFT_HAPI_DEF(af_ifft2_inplace)
FFT_HAPI_DEF(af_ifft3_inplace)

af_err af_fft(af_array *out, const af_array in, const double norm_factor, const dim_t odim0)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, norm_factor, odim0);
}

af_err af_fft2(af_array *out, const af_array in, const double norm_factor, const dim_t odim0, const dim_t odim1)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, norm_factor, odim0, odim1);
}

af_err af_fft3(af_array *out, const af_array in, const double norm_factor, const dim_t odim0, const dim_t odim1, const dim_t odim2)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, norm_factor, odim0, odim1, odim2);
}

af_err af_ifft(af_array *out, const af_array in, const double norm_factor, const dim_t odim0)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, norm_factor, odim0);
}

af_err af_ifft2(af_array *out, const af_array in, const double norm_factor, const dim_t odim0, const dim_t odim1)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, norm_factor, odim0, odim1);
}

af_err af_ifft3(af_array *out, const af_array in, const double norm_factor, const dim_t odim0, const dim_t odim1, const dim_t odim2)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, norm_factor, odim0, odim1, odim2);
}

af_err af_fft_r2c (af_array *out, const af_array in, const double norm_factor, const dim_t pad0)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, norm_factor, pad0);
}

af_err af_fft2_r2c(af_array *out, const af_array in, const double norm_factor, const dim_t pad0, const dim_t pad1)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, norm_factor, pad0, pad1);
}

af_err af_fft3_r2c(af_array *out, const af_array in, const double norm_factor, const dim_t pad0, const dim_t pad1, const dim_t pad2)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, norm_factor, pad0, pad1, pad2);
}

#define FFTC2R_HAPI_DEF(af_func)\
af_err af_func(af_array *out, const af_array in, const double norm_factor, const bool is_odd)\
{\
    CHECK_ARRAYS(in); \
    return CALL(out, in, norm_factor, is_odd);\
}

FFTC2R_HAPI_DEF(af_fft_c2r)
FFTC2R_HAPI_DEF(af_fft2_c2r)
FFTC2R_HAPI_DEF(af_fft3_c2r)

#define CONV_HAPI_DEF(af_func)\
af_err af_func(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode, af_conv_domain domain)\
{\
    CHECK_ARRAYS(signal, filter); \
    return CALL(out, signal, filter, mode, domain);\
}

CONV_HAPI_DEF(af_convolve1)
CONV_HAPI_DEF(af_convolve2)
CONV_HAPI_DEF(af_convolve3)

#define FFT_CONV_HAPI_DEF(af_func)\
af_err af_func(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode)\
{\
    CHECK_ARRAYS(signal, filter); \
    return CALL(out, signal, filter, mode);\
}

FFT_CONV_HAPI_DEF(af_fft_convolve1)
FFT_CONV_HAPI_DEF(af_fft_convolve2)
FFT_CONV_HAPI_DEF(af_fft_convolve3)

af_err af_convolve2_sep(af_array *out, const af_array col_filter, const af_array row_filter, const af_array signal, const af_conv_mode mode)
{
    CHECK_ARRAYS(col_filter, row_filter, signal);
    return CALL(out, col_filter, row_filter, signal, mode);
}

af_err af_fir(af_array *y, const af_array b, const af_array x)
{
    CHECK_ARRAYS(b, x);
    return CALL(y, b, x);
}

af_err af_iir(af_array *y, const af_array b, const af_array a, const af_array x)
{
    CHECK_ARRAYS(b, a, x);
    return CALL(y, b, a, x);
}
