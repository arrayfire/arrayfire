/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>

#ifdef __cplusplus

namespace af
{
class array;
class dim4;

/**
   C++ Interface for data interpolation on one dimensional data

   \param[in]  in is the input array
   \param[in]  pos array contains the interpolation locations
   \param[in]  method is the interpolation type, it can take one of the values defined by the
               enum \ref af_interp_type
   \param[in]  offGrid is the value that will set in the output array when certain index is out of bounds
   \return     the array with interpolated values

   \ingroup signal_func_approx1
 */
AFAPI array approx1(const array &in, const array &pos,
                    const interpType method = AF_INTERP_LINEAR, const float offGrid = 0.0f);

/**
   C++ Interface for data interpolation on two dimensional data

   \param[in]  in is the input array
   \param[in]  pos0 array contains the interpolation locations for first dimension
   \param[in]  pos1 array contains the interpolation locations for second dimension
   \param[in]  method is the interpolation type, it can take one of the values defined by the
               enum \ref af_interp_type
   \param[in]  offGrid is the value that will set in the output array when certain index is out of bounds
   \return     the array with interpolated values

   \ingroup signal_func_approx2
 */
AFAPI array approx2(const array &in, const array &pos0, const array &pos1,
                    const interpType method = AF_INTERP_LINEAR, const float offGrid = 0.0f);

/**
   C++ Interface for fast fourier transform on one dimensional data

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data - used to either truncate or pad the input data
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array fftNorm(const array& in, const double norm_factor, const dim_t odim0=0);

/**
   C++ Interface for fast fourier transform on two dimensional data

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along second dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_fft2
 */
AFAPI array fft2Norm(const array& in, const double norm_factor, const dim_t odim0=0, const dim_t odim1=0);

/**
   C++ Interface for fast fourier transform on three dimensional data

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along second dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output data along third dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_fft3
 */
AFAPI array fft3Norm(const array& in, const double norm_factor, const dim_t odim0=0, const dim_t odim1=0, const dim_t odim2=0);

/**
   C++ Interface for fast fourier transform on one dimensional data

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output data - used to either truncate or pad the input data
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array fft(const array& in, const dim_t odim0=0);

/**
   C++ Interface for fast fourier transform on two dimensional data

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output data along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along second dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_fft2
 */
AFAPI array fft2(const array& in, const dim_t odim0=0, const dim_t odim1=0);

/**
   C++ Interface for fast fourier transform on three dimensional data

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output data along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along second dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output data along third dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_fft3
 */
AFAPI array fft3(const array& in, const dim_t odim0=0, const dim_t odim1=0, const dim_t odim2=0);

/**
   C++ Interface for fast fourier transform on any(1d, 2d, 3d) dimensional data

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  outDims is an object of \ref dim4 that has the output array dimensions - used to either truncate or pad the input data
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array dft(const array& in, const double norm_factor, const dim4 outDims);

/**
   C++ Interface for fast fourier transform on any(1d, 2d, 3d) dimensional data

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \param[in]  outDims is an object of \ref dim4 that has the output array dimensions - used to either truncate or pad the input data
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array dft(const array& in, const dim4 outDims);

/**
   C++ Interface for fast fourier transform on any(1d, 2d, 3d) dimensional data

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array dft(const array& in);

/**
   C++ Interface for inverse fast fourier transform on one dimensional data

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data - used to either truncate or pad the input data
   \return     the transformed array

   \ingroup signal_func_ifft
 */
AFAPI array ifftNorm(const array& in, const double norm_factor, const dim_t odim0=0);

/**
   C++ Interface for inverse fast fourier transform on two dimensional data

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along second dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_ifft2
 */
AFAPI array ifft2Norm(const array& in, const double norm_factor, const dim_t odim0=0, const dim_t odim1=0);

/**
   C++ Interface for inverse fast fourier transform on three dimensional data

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along second dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output data along third dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_ifft3
 */
AFAPI array ifft3Norm(const array& in, const double norm_factor, const dim_t odim0=0, const dim_t odim1=0, const dim_t odim2=0);

/**
   C++ Interface for inverse fast fourier transform on one dimensional data

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output data - used to either truncate or pad the input data
   \return     the transformed array

   \ingroup signal_func_ifft
 */
AFAPI array ifft(const array& in, const dim_t odim0=0);

/**
   C++ Interface for inverse fast fourier transform on two dimensional data

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output data along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along second dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_ifft2
 */
AFAPI array ifft2(const array& in, const dim_t odim0=0, const dim_t odim1=0);

/**
   C++ Interface for inverse fast fourier transform on three dimensional data

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output data along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along second dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output data along third dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_ifft3
 */
AFAPI array ifft3(const array& in, const dim_t odim0=0, const dim_t odim1=0, const dim_t odim2=0);

/**
   C++ Interface for inverse fast fourier transform on any(1d, 2d, 3d) dimensional data

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  outDims is an object of \ref dim4 that has the output array dimensions - used to either truncate or pad the input data
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array idft(const array& in, const double norm_factor, const dim4 outDims);

/**
   C++ Interface for inverse fast fourier transform on any(1d, 2d, 3d) dimensional data

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \param[in]  outDims is an object of \ref dim4 that has the output array dimensions - used to either truncate or pad the input data
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array idft(const array& in, const dim4 outDims);

/**
   C++ Interface for inverse fast fourier transform on any(1d, 2d, 3d) dimensional data

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array idft(const array& in);

/**
   C++ Interface for convolution any(one through three) dimensional data

   Example for convolution on one dimensional signal in one to one batch mode
   \snippet test/convolve.cpp ex_image_convolve_1d

   Example for convolution on two dimensional signal in one to one batch mode
   \snippet test/convolve.cpp ex_image_convolve_2d

   Example for convolution on three dimensional signal in one to one batch mode
   \snippet test/convolve.cpp ex_image_convolve_3d

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \param[in]  domain specifies if the convolution should be performed in frequency os spatial domain
   \return     the convolved array

   \note The default parameter of \p domain, \ref AF_CONV_AUTO, heuristically switches between frequency and spatial domain.

   \ingroup signal_func_convolve
 */
AFAPI array convolve(const array& signal, const array& filter, const convMode mode=AF_CONV_DEFAULT, const convDomain domain=AF_CONV_AUTO);

/**
   C++ Interface for separable convolution on two dimensional data

   \snippet test/convolve.cpp ex_image_conv2_sep

   \param[in]  signal is the input signal
   \param[in]  col_filter is the signal that shall be along coloumns
   \param[in]  row_filter is the signal that shall be along rows
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     the convolved array

   \note The default parameter of \p domain, \ref AF_CONV_AUTO, heuristically switches between frequency and spatial domain.

   \note Separable convolution only supports two(ONE-to-ONE and MANY-to-ONE) batch modes from the ones described in the detailed description section.

   \ingroup signal_func_convolve
 */
AFAPI array convolve(const array& col_filter, const array& row_filter, const array& signal, const convMode mode=AF_CONV_DEFAULT);

/**
   C++ Interface for convolution on one dimensional data

   \snippet test/convolve.cpp ex_image_convolve1

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \param[in]  domain specifies if the convolution should be performed in frequency os spatial domain
   \return     the convolved array

   \note The default parameter of \p domain, \ref AF_CONV_AUTO, heuristically switches between frequency and spatial domain.

   \ingroup signal_func_convolve1
 */
AFAPI array convolve1(const array& signal, const array& filter, const convMode mode=AF_CONV_DEFAULT, const convDomain domain=AF_CONV_AUTO);

/**
   C++ Interface for convolution on two dimensional data

   \snippet test/convolve.cpp ex_image_convolve2

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \param[in]  domain specifies if the convolution should be performed in frequency os spatial domain
   \return     the convolved array

   \note The default parameter of \p domain, \ref AF_CONV_AUTO, heuristically switches between frequency and spatial domain.

   \ingroup signal_func_convolve2
 */
AFAPI array convolve2(const array& signal, const array& filter, const convMode mode=AF_CONV_DEFAULT, const convDomain domain=AF_CONV_AUTO);

/**
   C++ Interface for convolution on three dimensional data

   \snippet test/convolve.cpp ex_image_convolve3

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \param[in]  domain specifies if the convolution should be performed in frequency os spatial domain
   \return     the convolved array

   \note The default parameter of \p domain, \ref AF_CONV_AUTO, heuristically switches between frequency and spatial domain.

   \ingroup signal_func_convolve3
 */
AFAPI array convolve3(const array& signal, const array& filter, const convMode mode=AF_CONV_DEFAULT, const convDomain domain=AF_CONV_AUTO);

/**
   C++ Interface for FFT-based convolution any(one through three) dimensional data

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     the convolved array

   \ingroup signal_func_fft_convolve
 */
AFAPI array fftConvolve(const array& signal, const array& filter, const convMode mode=AF_CONV_DEFAULT);

/**
   C++ Interface for convolution on one dimensional data

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     the convolved array

   \ingroup signal_func_fft_convolve1
 */
AFAPI array fftConvolve1(const array& signal, const array& filter, const convMode mode=AF_CONV_DEFAULT);

/**
   C++ Interface for convolution on two dimensional data

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     the convolved array

   \ingroup signal_func_fft_convolve2
 */
AFAPI array fftConvolve2(const array& signal, const array& filter, const convMode mode=AF_CONV_DEFAULT);

/**
   C++ Interface for convolution on three dimensional data

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     the convolved array

   \ingroup signal_func_fftconvolve3
 */
AFAPI array fftConvolve3(const array& signal, const array& filter, const convMode mode=AF_CONV_DEFAULT);

/**
   C++ Interface for finite impulse response  filter

   \param[in] b is the array containing the coefficients of the filter
   \param[in] x is the input signal to the filter
   \returns the output signal from the filter

   \ingroup signal_func_fir
*/
AFAPI array fir(const array &b, const array &x);

/**
   C++ Interface for infinite impulse response filter

   \param[in] b is the array containing the feedforward coefficients
   \param[in] a is the array containing the feedback coefficients
   \param[in] x is the input signal to the filter
   \returns the output signal from the filter

   \note The feedforward coefficients are currently limited to a length of 512

   \ingroup signal_func_iir
*/
AFAPI array iir(const array &b, const array &a, const array &x);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
   C Interface for data interpolation on one dimensional data

   \param[out] out is the array with interpolated values
   \param[in]  in is the input array
   \param[in]  pos array contains the interpolation locations
   \param[in]  method is the interpolation type, it can take one of the values defined by the
               enum \ref af_interp_type
   \param[in]  offGrid is the value that will set in the output array when certain index is out of bounds
   \return     \ref AF_SUCCESS if the interpolation operation is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_approx1
 */
AFAPI af_err af_approx1(af_array *out, const af_array in, const af_array pos,
                        const af_interp_type method, const float offGrid);

/**
   C Interface for data interpolation on two dimensional data

   \param[out] out is the array with interpolated values
   \param[in]  in is the input array
   \param[in]  pos0 array contains the interpolation locations for first dimension
   \param[in]  pos1 array contains the interpolation locations for second dimension
   \param[in]  method is the interpolation type, it can take one of the values defined by the
               enum \ref af_interp_type
   \param[in]  offGrid is the value that will set in the output array when certain index is out of bounds
   \return     \ref AF_SUCCESS if the interpolation operation is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_approx2
 */
AFAPI af_err af_approx2(af_array *out, const af_array in, const af_array pos0, const af_array pos1,
                        const af_interp_type method, const float offGrid);

/**
   C Interface for fast fourier transform on one dimensional data

   \param[out] out is the transformed array
   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data - used to either truncate or pad the input data
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_fft
 */
AFAPI af_err af_fft(af_array *out, const af_array in, const double norm_factor, const dim_t odim0);

/**
   C Interface for fast fourier transform on two dimensional data

   \param[out] out is the transformed array
   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along second dimension - used to either truncate/pad the input
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_fft2
 */
AFAPI af_err af_fft2(af_array *out, const af_array in, const double norm_factor, const dim_t odim0, const dim_t odim1);

/**
   C Interface for fast fourier transform on three dimensional data

   \param[out] out is the transformed array
   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along second dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output data along third dimension - used to either truncate/pad the input
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_fft3
 */
AFAPI af_err af_fft3(af_array *out, const af_array in, const double norm_factor, const dim_t odim0, const dim_t odim1, const dim_t odim2);

/**
   C Interface for inverse fast fourier transform on one dimensional data

   \param[out] out is the transformed array
   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data - used to either truncate or pad the input data
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_ifft
 */
AFAPI af_err af_ifft(af_array *out, const af_array in, const double norm_factor, const dim_t odim0);

/**
   C Interface for inverse fast fourier transform on two dimensional data

   \param[out] out is the transformed array
   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along second dimension - used to either truncate/pad the input
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_ifft2
 */
AFAPI af_err af_ifft2(af_array *out, const af_array in, const double norm_factor, const dim_t odim0, const dim_t odim1);

/**
   C Interface for inverse fast fourier transform on three dimensional data

   \param[out] out is the transformed array
   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along second dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output data along third dimension - used to either truncate/pad the input
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_ifft3
 */
AFAPI af_err af_ifft3(af_array *out, const af_array in, const double norm_factor, const dim_t odim0, const dim_t odim1, const dim_t odim2);

/**
   C Interface for convolution on one dimensional data

   \param[out] out is convolved array
   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \param[in]  domain specifies if the convolution should be performed in frequency os spatial domain
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \note The default parameter of \p domain, \ref AF_CONV_AUTO, heuristically switches between frequency and spatial domain.

   \ingroup signal_func_convolve1
 */
AFAPI af_err af_convolve1(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode, af_conv_domain domain);

/**
   C Interface for convolution on two dimensional data

   \param[out] out is convolved array
   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \param[in]  domain specifies if the convolution should be performed in frequency os spatial domain
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \note The default parameter of \p domain, \ref AF_CONV_AUTO, heuristically switches between frequency and spatial domain.

   \ingroup signal_func_convolve2
 */
AFAPI af_err af_convolve2(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode, af_conv_domain domain);

/**
   C Interface for convolution on three dimensional data

   \param[out] out is convolved array
   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \param[in]  domain specifies if the convolution should be performed in frequency os spatial domain
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \note The default parameter of \p domain, \ref AF_CONV_AUTO, heuristically switches between frequency and spatial domain.

   \ingroup signal_func_convolve3
 */
AFAPI af_err af_convolve3(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode, af_conv_domain domain);

/**
   C Interface for separable convolution on two dimensional data

   \param[out] out is convolved array
   \param[in]  col_filter is filter that has to be applied along the coloumns
   \param[in]  row_filter is filter that has to be applied along the rows
   \param[in]  signal is the input array
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \note Separable convolution only supports two(ONE-to-ONE and MANY-to-ONE) batch modes from the ones described
         in the detailed description section.

   \ingroup signal_func_convolve
 */
AFAPI af_err af_convolve2_sep(af_array *out, const af_array col_filter, const af_array row_filter, const af_array signal, const af_conv_mode mode);

/**
   C Interface for FFT-based convolution on one dimensional data

   \param[out] out is convolved array
   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_fft_convolve1
 */
AFAPI af_err af_fft_convolve1(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode);

/**
   C Interface for FFT-based convolution on two dimensional data

   \param[out] out is convolved array
   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_fft_convolve2
 */
AFAPI af_err af_fft_convolve2(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode);

/**
   C Interface for FFT-based convolution on three dimensional data

   \param[out] out is convolved array
   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_fft_convolve3
 */
AFAPI af_err af_fft_convolve3(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode);

/**
   C++ Interface for finite impulse response  filter

   \param[out] y is the output signal from the filter
   \param[in] b is the array containing the coefficients of the filter
   \param[in] x is the input signal to the filter

   \ingroup signal_func_fir
*/
AFAPI af_err af_fir(af_array *y, const af_array b, const af_array x);

/**
   C++ Interface for infinite impulse response filter

   \param[out] y is the output signal from the filter
   \param[in] b is the array containing the feedforward coefficients
   \param[in] a is the array containing the feedback coefficients
   \param[in] x is the input signal to the filter

   \note The feedforward coefficients are currently limited to a length of 512

   \ingroup signal_func_iir
*/
AFAPI af_err af_iir(af_array *y, const af_array b, const af_array a, const af_array x);
#ifdef __cplusplus
}
#endif
