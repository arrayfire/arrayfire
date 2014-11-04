/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/array.h>

#ifdef __cplusplus
namespace af
{

AFAPI array approx1(const array &in, const array &pos,
                    const af_interp_type method = AF_INTERP_LINEAR, const float offGrid = 0.0f);

AFAPI array approx2(const array &in, const array &pos0, const array &pos1,
                    const af_interp_type method = AF_INTERP_LINEAR, const float offGrid = 0.0f);

AFAPI array fft(const array& in, double normalize, dim_type odim0=0);

AFAPI array fft2(const array& in, double normalize, dim_type odim0=0, dim_type odim1=0);

AFAPI array fft3(const array& in, double normalize, dim_type odim0=0, dim_type odim1=0, dim_type odim2=0);

AFAPI array ifft(const array& in, double normalize, dim_type odim0=0);

AFAPI array ifft2(const array& in, double normalize, dim_type odim0=0, dim_type odim1=0);

AFAPI array ifft3(const array& in, double normalize, dim_type odim0=0, dim_type odim1=0, dim_type odim2=0);

AFAPI array fft(const array& in, dim_type odim0=0);

AFAPI array fft2(const array& in, dim_type odim0=0, dim_type odim1=0);

AFAPI array fft3(const array& in, dim_type odim0=0, dim_type odim1=0, dim_type odim2=0);

AFAPI array ifft(const array& in, dim_type odim0=0);

AFAPI array ifft2(const array& in, dim_type odim0=0, dim_type odim1=0);

AFAPI array ifft3(const array& in, dim_type odim0=0, dim_type odim1=0, dim_type odim2=0);


AFAPI array convolve(const array& signal, const array& filter, bool expand=true);

AFAPI array convolve(const array& signal, const array& col_filter, const array& row_filter, bool expand=true);

AFAPI array convolve1(const array& signal, const array& filter, bool expand=true);

AFAPI array convolve2(const array& signal, const array& filter, bool expand=true);

AFAPI array convolve3(const array& signal, const array& filter, bool expand=true);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Interpolation in 1D
AFAPI af_err af_approx1(af_array *out, const af_array in, const af_array pos,
                        const af_interp_type method, const float offGrid);

// Interpolation in 2D
AFAPI af_err af_approx2(af_array *out, const af_array in, const af_array pos0, const af_array pos1,
                        const af_interp_type method, const float offGrid);


AFAPI af_err af_fft(af_array *out, af_array in, double normalize, dim_type odim0);

AFAPI af_err af_fft2(af_array *out, af_array in, double normalize, dim_type odim0, dim_type odim1);

AFAPI af_err af_fft3(af_array *out, af_array in, double normalize, dim_type odim0, dim_type odim1, dim_type odim2);

AFAPI af_err af_ifft(af_array *out, af_array in, double normalize, dim_type odim0);

AFAPI af_err af_ifft2(af_array *out, af_array in, double normalize, dim_type odim0, dim_type odim1);

AFAPI af_err af_ifft3(af_array *out, af_array in, double normalize, dim_type odim0, dim_type odim1, dim_type odim2);


AFAPI af_err af_convolve1(af_array *out, af_array signal, af_array filter, bool expand);

AFAPI af_err af_convolve2(af_array *out, af_array signal, af_array filter, bool expand);

AFAPI af_err af_convolve3(af_array *out, af_array signal, af_array filter, bool expand);

AFAPI af_err af_convolve2_sep(af_array *out, af_array signal, af_array col_filter, af_array row_filter, bool expand);

#ifdef __cplusplus
}
#endif
