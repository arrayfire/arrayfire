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
   \param[in]  pos0 array contains the interpolation locations for 0th dimension
   \param[in]  pos1 array contains the interpolation locations for 1st dimension
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
AFAPI array fft(const array& in, double norm_factor, dim_type odim0=0);

/**
   C++ Interface for fast fourier transform on two dimensional data

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data along 0th dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along 1st dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_fft2
 */
AFAPI array fft2(const array& in, double norm_factor, dim_type odim0=0, dim_type odim1=0);

/**
   C++ Interface for fast fourier transform on three dimensional data

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data along 0th dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along 1st dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output data along 2nd dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_fft3
 */
AFAPI array fft3(const array& in, double norm_factor, dim_type odim0=0, dim_type odim1=0, dim_type odim2=0);

/**
   C++ Interface for fast fourier transform on one dimensional data

   This version of fft function uses a default norm_factor paramter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output data - used to either truncate or pad the input data
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array fft(const array& in, dim_type odim0=0);

/**
   C++ Interface for fast fourier transform on two dimensional data

   This version of fft function uses a default norm_factor paramter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output data along 0th dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along 1st dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_fft2
 */
AFAPI array fft2(const array& in, dim_type odim0=0, dim_type odim1=0);

/**
   C++ Interface for fast fourier transform on three dimensional data

   This version of fft function uses a default norm_factor paramter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output data along 0th dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along 1st dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output data along 2nd dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_fft3
 */
AFAPI array fft3(const array& in, dim_type odim0=0, dim_type odim1=0, dim_type odim2=0);

/**
   C++ Interface for fast fourier transform on any(1d, 2d, 3d) dimensional data

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  outDims is an object of \ref dim4 that has the output array dimensions - used to either truncate or pad the input data
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array dft(const array& in, double norm_factor, const dim4 outDims);

/**
   C++ Interface for fast fourier transform on any(1d, 2d, 3d) dimensional data

   This version of fft function uses a default norm_factor paramter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \param[in]  outDims is an object of \ref dim4 that has the output array dimensions - used to either truncate or pad the input data
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array dft(const array& in, const dim4 outDims);

/**
   C++ Interface for fast fourier transform on any(1d, 2d, 3d) dimensional data

   This version of fft function uses a default norm_factor paramter that is calculated internally
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
AFAPI array ifft(const array& in, double norm_factor, dim_type odim0=0);

/**
   C++ Interface for inverse fast fourier transform on two dimensional data

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data along 0th dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along 1st dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_ifft2
 */
AFAPI array ifft2(const array& in, double norm_factor, dim_type odim0=0, dim_type odim1=0);

/**
   C++ Interface for inverse fast fourier transform on three dimensional data

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data along 0th dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along 1st dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output data along 2nd dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_ifft3
 */
AFAPI array ifft3(const array& in, double norm_factor, dim_type odim0=0, dim_type odim1=0, dim_type odim2=0);

/**
   C++ Interface for inverse fast fourier transform on one dimensional data

   This version of fft function uses a default norm_factor paramter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output data - used to either truncate or pad the input data
   \return     the transformed array

   \ingroup signal_func_ifft
 */
AFAPI array ifft(const array& in, dim_type odim0=0);

/**
   C++ Interface for inverse fast fourier transform on two dimensional data

   This version of fft function uses a default norm_factor paramter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output data along 0th dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along 1st dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_ifft2
 */
AFAPI array ifft2(const array& in, dim_type odim0=0, dim_type odim1=0);

/**
   C++ Interface for inverse fast fourier transform on three dimensional data

   This version of fft function uses a default norm_factor paramter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output data along 0th dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along 1st dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output data along 2nd dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_ifft3
 */
AFAPI array ifft3(const array& in, dim_type odim0=0, dim_type odim1=0, dim_type odim2=0);

/**
   C++ Interface for inverse fast fourier transform on any(1d, 2d, 3d) dimensional data

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  outDims is an object of \ref dim4 that has the output array dimensions - used to either truncate or pad the input data
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array idft(const array& in, double norm_factor, const dim4 outDims);

/**
   C++ Interface for inverse fast fourier transform on any(1d, 2d, 3d) dimensional data

   This version of fft function uses a default norm_factor paramter that is calculated internally
   based on the input data.

   \param[in]  in is the input array
   \param[in]  outDims is an object of \ref dim4 that has the output array dimensions - used to either truncate or pad the input data
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array idft(const array& in, const dim4 outDims);

/**
   C++ Interface for inverse fast fourier transform on any(1d, 2d, 3d) dimensional data

   This version of fft function uses a default norm_factor paramter that is calculated internally
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
   \param[in]  expand indicates if the convolution should be expanded or not(where output size equals input).
   \return     the convolved array

   \ingroup signal_func_convolve
 */
AFAPI array convolve(const array& signal, const array& filter, bool expand=false);

/**
   C++ Interface for separable convolution on two dimensional data

   \snippet test/convolve.cpp ex_image_conv2_sep

   \param[in]  signal is the input signal
   \param[in]  col_filter is the signal that shall be along coloumns
   \param[in]  row_filter is the signal that shall be along rows
   \param[in]  expand indicates if the convolution should be expanded or not(where output size equals input).
   \return     the convolved array

   \note Separable convolution only supports two(ONE-to-ONE and MANY-to-ONE) batch modes from the ones described
         in the detailed description section.

   \ingroup signal_func_convolve
 */
AFAPI array convolve(const array& col_filter, const array& row_filter, const array& signal, bool expand=false);

/**
   C++ Interface for convolution on one dimensional data

   \snippet test/convolve.cpp ex_image_convolve1

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  expand indicates if the convolution should be expanded or not(where output size equals input).
   \return     the convolved array

   \ingroup signal_func_convolve1
 */
AFAPI array convolve1(const array& signal, const array& filter, bool expand=false);

/**
   C++ Interface for convolution on two dimensional data

   \snippet test/convolve.cpp ex_image_convolve2

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  expand indicates if the convolution should be expanded or not(where output size equals input).
   \return     the convolved array

   \ingroup signal_func_convolve2
 */
AFAPI array convolve2(const array& signal, const array& filter, bool expand=false);

/**
   C++ Interface for convolution on three dimensional data

   \snippet test/convolve.cpp ex_image_convolve3

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  expand indicates if the convolution should be expanded or not(where output size equals input).
   \return     the convolved array

   \ingroup signal_func_convolve3
 */
AFAPI array convolve3(const array& signal, const array& filter, bool expand=false);

/**
   C++ Interface for FFT-based convolution any(one through three) dimensional data

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \return     the convolved array

   \ingroup signal_func_fftconvolve
 */
AFAPI array fftconvolve(const array& signal, const array& filter);

/**
   C++ Interface for convolution on one dimensional data

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \return     the convolved array

   \ingroup signal_func_fftconvolve1
 */
AFAPI array fftconvolve1(const array& signal, const array& filter);

/**
   C++ Interface for convolution on two dimensional data

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \return     the convolved array

   \ingroup signal_func_fftconvolve2
 */
AFAPI array fftconvolve2(const array& signal, const array& filter);

/**
   C++ Interface for convolution on three dimensional data

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \return     the convolved array

   \ingroup signal_func_fftconvolve3
 */
AFAPI array fftconvolve3(const array& signal, const array& filter);

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
   \param[in]  pos0 array contains the interpolation locations for 0th dimension
   \param[in]  pos1 array contains the interpolation locations for 1st dimension
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
AFAPI af_err af_fft(af_array *out, af_array in, double norm_factor, dim_type odim0);

/**
   C Interface for fast fourier transform on two dimensional data

   \param[out] out is the transformed array
   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data along 0th dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along 1st dimension - used to either truncate/pad the input
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_fft2
 */
AFAPI af_err af_fft2(af_array *out, af_array in, double norm_factor, dim_type odim0, dim_type odim1);

/**
   C Interface for fast fourier transform on three dimensional data

   \param[out] out is the transformed array
   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data along 0th dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along 1st dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output data along 2nd dimension - used to either truncate/pad the input
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_fft3
 */
AFAPI af_err af_fft3(af_array *out, af_array in, double norm_factor, dim_type odim0, dim_type odim1, dim_type odim2);

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
AFAPI af_err af_ifft(af_array *out, af_array in, double norm_factor, dim_type odim0);

/**
   C Interface for inverse fast fourier transform on two dimensional data

   \param[out] out is the transformed array
   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data along 0th dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along 1st dimension - used to either truncate/pad the input
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_ifft2
 */
AFAPI af_err af_ifft2(af_array *out, af_array in, double norm_factor, dim_type odim0, dim_type odim1);

/**
   C Interface for inverse fast fourier transform on three dimensional data

   \param[out] out is the transformed array
   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output data along 0th dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output data along 1st dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output data along 2nd dimension - used to either truncate/pad the input
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_ifft3
 */
AFAPI af_err af_ifft3(af_array *out, af_array in, double norm_factor, dim_type odim0, dim_type odim1, dim_type odim2);

/**
   C Interface for convolution on one dimensional data

   \param[out] out is convolved array
   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be flipped for the convolution operation
   \param[in]  expand indicates if the convolution should be expanded or not(where output size equals input).
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_convolve1
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

   \ingroup signal_func_convolve2
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

   \ingroup signal_func_convolve3
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

   \note Separable convolution only supports two(ONE-to-ONE and MANY-to-ONE) batch modes from the ones described
         in the detailed description section.

   \ingroup signal_func_convolve
 */
AFAPI af_err af_convolve2_sep(af_array *out, af_array col_filter, af_array row_filter, af_array signal, bool expand);

/**
   C Interface for FFT-based convolution on one dimensional data

   \param[out] out is convolved array
   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_fftconvolve1
 */
AFAPI af_err af_fftconvolve1(af_array *out, af_array signal, af_array filter);

/**
   C Interface for FFT-based convolution on two dimensional data

   \param[out] out is convolved array
   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_fftconvolve2
 */
AFAPI af_err af_fftconvolve2(af_array *out, af_array signal, af_array filter);

/**
   C Interface for FFT-based convolution on three dimensional data

   \param[out] out is convolved array
   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_fftconvolve3
 */
AFAPI af_err af_fftconvolve3(af_array *out, af_array signal, af_array filter);

#ifdef __cplusplus
}
#endif
