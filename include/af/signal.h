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
                    const interpType method = AF_INTERP_LINEAR, const float offGrid = 0.0f);

AFAPI array approx2(const array &in, const array &pos0, const array &pos1,
                    const interpType method = AF_INTERP_LINEAR, const float offGrid = 0.0f);

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


/**
   C++ Interface for convolution any(one through three) dimensional data

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  expand indicates if the convolution should be expanded or not(where output size equals input).
   \return     the convolved array

   \ingroup signal_func_convolve
 */
AFAPI array convolve(const array& signal, const array& filter, bool expand=false);

/**
   C++ Interface for separable convolution on two dimensional data

   \param[in]  signal is the input signal
   \param[in]  col_filter is the signal that shall be along coloumns
   \param[in]  row_filter is the signal that shall be along rows
   \param[in]  expand indicates if the convolution should be expanded or not(where output size equals input).
   \return     the convolved array

   \ingroup signal_func_convolve
 */
AFAPI array convolve(const array& col_filter, const array& row_filter, const array& signal, bool expand=false);

/**
   C++ Interface for convolution on one dimensional data

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  expand indicates if the convolution should be expanded or not(where output size equals input).
   \return     the convolved array

   \ingroup signal_func_convolve
 */
AFAPI array convolve1(const array& signal, const array& filter, bool expand=false);

/**
   C++ Interface for convolution on two dimensional data

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  expand indicates if the convolution should be expanded or not(where output size equals input).
   \return     the convolved array

   \ingroup signal_func_convolve
 */
AFAPI array convolve2(const array& signal, const array& filter, bool expand=false);

/**
   C++ Interface for convolution on three dimensional data

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  expand indicates if the convolution should be expanded or not(where output size equals input).
   \return     the convolved array

   \ingroup signal_func_convolve
 */
AFAPI array convolve3(const array& signal, const array& filter, bool expand=false);

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


/**
   C Interface for convolution on one dimensional data

   \param[out] out is convolved array
   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  expand indicates if the convolution should be expanded or not(where output size equals input).
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_convolve
 */
AFAPI af_err af_convolve1(af_array *out, af_array signal, af_array filter, bool expand);

/**
   C Interface for convolution on two dimensional data

   \param[out] out is convolved array
   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  expand indicates if the convolution should be expanded or not(where output size equals input).
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_convolve
 */
AFAPI af_err af_convolve2(af_array *out, af_array signal, af_array filter, bool expand);

/**
   C Interface for convolution on three dimensional data

   \param[out] out is convolved array
   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  expand indicates if the convolution should be expanded or not(where output size equals input).
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_convolve
 */
AFAPI af_err af_convolve3(af_array *out, af_array signal, af_array filter, bool expand);

/**
   C Interface for separable convolution on two dimensional data

   \param[out] out is convolved array
   \param[in]  col_filter is filter that has to be applied along the coloumns
   \param[in]  row_filter is filter that has to be applied along the rows
   \param[in]  signal is the input array
   \param[in]  expand indicates if the convolution should be expanded or not(where output size equals input).
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_convolve
 */
AFAPI af_err af_convolve2_sep(af_array *out, af_array col_filter, af_array row_filter, af_array signal, bool expand);

#ifdef __cplusplus
}
#endif
