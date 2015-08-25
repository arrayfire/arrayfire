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
   C++ Interface for data interpolation on one dimensional signals

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
   C++ Interface for data interpolation on two dimensional signals

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
   C++ Interface for fast fourier transform on one dimensional signals

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output signals - used to either truncate or pad the input signals
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array fftNorm(const array& in, const double norm_factor, const dim_t odim0=0);

/**
   C++ Interface for fast fourier transform on two dimensional signals

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output signals along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output signals along second dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_fft2
 */
AFAPI array fft2Norm(const array& in, const double norm_factor, const dim_t odim0=0, const dim_t odim1=0);

/**
   C++ Interface for fast fourier transform on three dimensional signals

   \param[in]  in is the input array and the output of 1D fourier transform on exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output signals along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output signals along second dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output signals along third dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_fft3
 */
AFAPI array fft3Norm(const array& in, const double norm_factor, const dim_t odim0=0, const dim_t odim1=0, const dim_t odim2=0);

#if AF_API_VERSION >= 31
/**
   C++ Interface for fast fourier transform on one dimensional signals

   \param[inout]  in is the input array on entry and the output of 1D forward fourier transform on exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied

   \note The input \p in must be complex

   \ingroup signal_func_fft
 */
AFAPI void fftInPlace(array& in, const double norm_factor = 1);
#endif

#if AF_API_VERSION >= 31
/**
   C++ Interface for fast fourier transform on two dimensional signals

   \param[inout]  in is the input array on entry and the output of 2D forward fourier transform on exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \return     the transformed array

   \note The input \p in must be complex

   \ingroup signal_func_fft2
 */
AFAPI void fft2InPlace(array& in, const double norm_factor = 1);
#endif

#if AF_API_VERSION >= 31
/**
   C++ Interface for fast fourier transform on three dimensional signals

   \param[inout]  in is the input array on entry and the output of 3D forward fourier transform on exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \return     the transformed array

   \note The input \p in must be complex

   \ingroup signal_func_fft3
 */
AFAPI void fft3InPlace(array& in, const double norm_factor = 1);
#endif

/**
   C++ Interface for fast fourier transform on one dimensional signals

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input signals.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output signals - used to either truncate or pad the input signals
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array fft(const array& in, const dim_t odim0=0);

/**
   C++ Interface for fast fourier transform on two dimensional signals

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input signals.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output signals along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output signals along second dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_fft2
 */
AFAPI array fft2(const array& in, const dim_t odim0=0, const dim_t odim1=0);

/**
   C++ Interface for fast fourier transform on three dimensional signals

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input signals.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output signals along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output signals along second dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output signals along third dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_fft3
 */
AFAPI array fft3(const array& in, const dim_t odim0=0, const dim_t odim1=0, const dim_t odim2=0);

/**
   C++ Interface for fast fourier transform on any(1d, 2d, 3d) dimensional signals

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  outDims is an object of \ref dim4 that has the output array dimensions - used to either truncate or pad the input signals
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array dft(const array& in, const double norm_factor, const dim4 outDims);

/**
   C++ Interface for fast fourier transform on any(1d, 2d, 3d) dimensional signals

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input signals.

   \param[in]  in is the input array
   \param[in]  outDims is an object of \ref dim4 that has the output array dimensions - used to either truncate or pad the input signals
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array dft(const array& in, const dim4 outDims);

/**
   C++ Interface for fast fourier transform on any(1d, 2d, 3d) dimensional signals

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input signals.

   \param[in]  in is the input array
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array dft(const array& in);

/**
   C++ Interface for inverse fast fourier transform on one dimensional signals

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output signals - used to either truncate or pad the input signals
   \return     the transformed array

   \ingroup signal_func_ifft
 */
AFAPI array ifftNorm(const array& in, const double norm_factor, const dim_t odim0=0);

/**
   C++ Interface for inverse fast fourier transform on two dimensional signals

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output signals along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output signals along second dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_ifft2
 */
AFAPI array ifft2Norm(const array& in, const double norm_factor, const dim_t odim0=0, const dim_t odim1=0);

/**
   C++ Interface for inverse fast fourier transform on three dimensional signals

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output signals along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output signals along second dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output signals along third dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_ifft3
 */
AFAPI array ifft3Norm(const array& in, const double norm_factor, const dim_t odim0=0, const dim_t odim1=0, const dim_t odim2=0);

#if AF_API_VERSION >= 31
/**
   C++ Interface for fast fourier transform on one dimensional signals

   \param[inout]  in is the input array on entry and the output of 1D inverse fourier transform on exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied

   \note The input \p in must be complex

   \ingroup signal_func_ifft
 */
AFAPI void ifftInPlace(array& in, const double norm_factor = 1);
#endif

#if AF_API_VERSION >= 31
/**
   C++ Interface for fast fourier transform on two dimensional signals

   \param[inout]  in is the input array on entry and the output of 2D inverse fourier transform on exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \return     the transformed array

   \note The input \p in must be complex

   \ingroup signal_func_ifft2
 */
AFAPI void ifft2InPlace(array& in, const double norm_factor = 1);
#endif

#if AF_API_VERSION >= 31
/**
   C++ Interface for fast fourier transform on three dimensional signals

   \param[inout]  in is the input array on entry and the output of 3D inverse fourier transform on exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \return     the transformed array

   \note The input \p in must be complex

   \ingroup signal_func_ifft3
 */
AFAPI void ifft3InPlace(array& in, const double norm_factor = 1);
#endif

/**
   C++ Interface for inverse fast fourier transform on one dimensional signals

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input signals.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output signals - used to either truncate or pad the input signals
   \return     the transformed array

   \ingroup signal_func_ifft
 */
AFAPI array ifft(const array& in, const dim_t odim0=0);

/**
   C++ Interface for inverse fast fourier transform on two dimensional signals

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input signals.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output signals along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output signals along second dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_ifft2
 */
AFAPI array ifft2(const array& in, const dim_t odim0=0, const dim_t odim1=0);

/**
   C++ Interface for inverse fast fourier transform on three dimensional signals

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input signals.

   \param[in]  in is the input array
   \param[in]  odim0 is the length of output signals along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output signals along second dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output signals along third dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_ifft3
 */
AFAPI array ifft3(const array& in, const dim_t odim0=0, const dim_t odim1=0, const dim_t odim2=0);

/**
   C++ Interface for inverse fast fourier transform on any(1d, 2d, 3d) dimensional signals

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  outDims is an object of \ref dim4 that has the output array dimensions - used to either truncate or pad the input signals
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array idft(const array& in, const double norm_factor, const dim4 outDims);

/**
   C++ Interface for inverse fast fourier transform on any(1d, 2d, 3d) dimensional signals

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input signals.

   \param[in]  in is the input array
   \param[in]  outDims is an object of \ref dim4 that has the output array dimensions - used to either truncate or pad the input signals
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array idft(const array& in, const dim4 outDims);

/**
   C++ Interface for inverse fast fourier transform on any(1d, 2d, 3d) dimensional signals

   This version of fft function uses a default norm_factor parameter that is calculated internally
   based on the input signals.

   \param[in]  in is the input array
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array idft(const array& in);

#if AF_API_VERSION >= 31
/**
   C++ Interface for real to complex fast fourier transform for one dimensional signals

   \param[in]  in is a real array
   \param[in]  dims is the requested padded dimensions before the transform is applied
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \return     a complex array containing the non redundant parts of \p in along the first dimension.

   \note The first dimension of the output will be of size (dims[0] / 2) + 1. The remaining dimensions are unchanged.

   \ingroup signal_func_fft_r2c
*/
template<int rank>
array fftR2C(const array &in,
             const dim4& dims,
             const double norm_factor = 0);
#endif

#if AF_API_VERSION >= 31
/**
   C++ Interface for real to complex fast fourier transform for one dimensional signals

   \param[in]  in is a real array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \return     a complex array containing the non redundant parts of \p in along the first dimension.

   \note The first dimension of the output will be of size (in.dims(0) / 2) + 1. The remaining dimensions are unchanged.

   \ingroup signal_func_fft_r2c
*/
template<int rank>
array fftR2C(const array &in,
             const double norm_factor = 0);
#endif

#if AF_API_VERSION >= 31
/**
   C++ Interface for complex to real fast fourier transform

   \param[in]  in is a complex array containing only the non redundant parts of the signals
   \param[in]  is_odd is a flag signifying if the output should be even or odd size
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \tparam     rank signifies the dimensionality of the transform
   \return     A real array of size [2 * idim0 - 2 + is_odd, idim1, idim2, idim3] where idim{0,1,2,3} signify input dimensions

   \ingroup signal_func_fft_c2r
*/

template<int rank>
array fftC2R(const array &in, bool is_odd = false,
                 const double norm_factor = 0);
#endif

/**
   C++ Interface for convolution any(one through three) dimensional signals

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
   C++ Interface for separable convolution on two dimensional signals

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
   C++ Interface for convolution on one dimensional signals

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
   C++ Interface for convolution on two dimensional signals

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
   C++ Interface for convolution on three dimensional signals

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
   C++ Interface for FFT-based convolution any(one through three) dimensional signals

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     the convolved array

   \ingroup signal_func_fft_convolve
 */
AFAPI array fftConvolve(const array& signal, const array& filter, const convMode mode=AF_CONV_DEFAULT);

/**
   C++ Interface for convolution on one dimensional signals

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     the convolved array

   \ingroup signal_func_fft_convolve1
 */
AFAPI array fftConvolve1(const array& signal, const array& filter, const convMode mode=AF_CONV_DEFAULT);

/**
   C++ Interface for convolution on two dimensional signals

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     the convolved array

   \ingroup signal_func_fft_convolve2
 */
AFAPI array fftConvolve2(const array& signal, const array& filter, const convMode mode=AF_CONV_DEFAULT);

/**
   C++ Interface for convolution on three dimensional signals

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
   C Interface for signals interpolation on one dimensional signals

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
   C Interface for signals interpolation on two dimensional signals

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
   C Interface for fast fourier transform on one dimensional signals

   \param[out] out is the transformed array
   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output signals - used to either truncate or pad the input signals
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_fft
 */
AFAPI af_err af_fft(af_array *out, const af_array in, const double norm_factor, const dim_t odim0);

#if AF_API_VERSION >= 31
/**
   C Interface for fast fourier transform on one dimensional signals

   \param[inout]  in is the input array on entry and the output of 1D forward fourier transform at exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \note The input \p in must be a complex array

   \ingroup signal_func_fft
*/
AFAPI af_err af_fft_inplace(af_array in, const double norm_factor);
#endif

/**
   C Interface for fast fourier transform on two dimensional signals

   \param[out] out is the transformed array
   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output signals along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output signals along second dimension - used to either truncate/pad the input
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_fft2
 */
AFAPI af_err af_fft2(af_array *out, const af_array in, const double norm_factor, const dim_t odim0, const dim_t odim1);

#if AF_API_VERSION >= 31
/**
   C Interface for fast fourier transform on two dimensional signals

   \param[inout]  in is the input array on entry and the output of 2D forward fourier transform on exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \note The input \p in must be a complex array

   \ingroup signal_func_fft2
 */
AFAPI af_err af_fft2_inplace(af_array in, const double norm_factor);
#endif

/**
   C Interface for fast fourier transform on three dimensional signals

   \param[out] out is the transformed array
   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output signals along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output signals along second dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output signals along third dimension - used to either truncate/pad the input
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_fft3
 */
AFAPI af_err af_fft3(af_array *out, const af_array in, const double norm_factor, const dim_t odim0, const dim_t odim1, const dim_t odim2);

#if AF_API_VERSION >= 31
/**
   C Interface for fast fourier transform on three dimensional signals

   \param[inout]  in is the input array on entry and the output of 3D forward fourier transform on exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \note The input \p must be a complex array

   \ingroup signal_func_fft3
 */
AFAPI af_err af_fft3_inplace(af_array in, const double norm_factor);
#endif

/**
   C Interface for inverse fast fourier transform on one dimensional signals

   \param[out] out is the transformed array
   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output signals - used to either truncate or pad the input signals
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_ifft
 */
AFAPI af_err af_ifft(af_array *out, const af_array in, const double norm_factor, const dim_t odim0);

#if AF_API_VERSION >= 31
/**
   C Interface for fast fourier transform on one dimensional signals

   \param[inout]  in is the input array on entry and the output of 1D inverse fourier transform at exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \return     \ref AF_SUCCESS if the ifft transform is successful,
               otherwise an appropriate error code is returned.

   \note The input \p in must be a complex array

   \ingroup signal_func_ifft
*/
AFAPI af_err af_ifft_inplace(af_array in, const double norm_factor);
#endif

/**
   C Interface for inverse fast fourier transform on two dimensional signals

   \param[out] out is the transformed array
   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output signals along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output signals along second dimension - used to either truncate/pad the input
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_ifft2
 */
AFAPI af_err af_ifft2(af_array *out, const af_array in, const double norm_factor, const dim_t odim0, const dim_t odim1);

#if AF_API_VERSION >= 31
/**
   C Interface for fast fourier transform on two dimensional signals

   \param[inout]  in is the input array on entry and the output of 2D inverse fourier transform on exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \return     \ref AF_SUCCESS if the ifft transform is successful,
               otherwise an appropriate error code is returned.

   \note The input \p in must be a complex array

   \ingroup signal_func_ifft2
*/
AFAPI af_err af_ifft2_inplace(af_array in, const double norm_factor);
#endif

/**
   C Interface for inverse fast fourier transform on three dimensional signals

   \param[out] out is the transformed array
   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  odim0 is the length of output signals along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output signals along second dimension - used to either truncate/pad the input
   \param[in]  odim2 is the length of output signals along third dimension - used to either truncate/pad the input
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_ifft3
 */
AFAPI af_err af_ifft3(af_array *out, const af_array in, const double norm_factor, const dim_t odim0, const dim_t odim1, const dim_t odim2);

#if AF_API_VERSION >= 31
/**
   C Interface for fast fourier transform on three dimensional signals

   \param[inout]  in is the input array on entry and the output of 3D inverse fourier transform on exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \return     \ref AF_SUCCESS if the ifft transform is successful,
               otherwise an appropriate error code is returned.

   \note The input \p must be a complex array

   \ingroup signal_func_ifft3
*/
AFAPI af_err af_ifft3_inplace(af_array in, const double norm_factor);
#endif

#if AF_API_VERSION >= 31
/**
   C Interface for real to complex fast fourier transform for one dimensional signals

   \param[out] out is a complex array containing the non redundant parts of \p in.
   \param[in]  in is a real array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  pad0 is the length of output signals along first dimension - used to either truncate/pad the input
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \note The first dimension of the output will be of size (pad0 / 2) + 1. The remaining dimensions are unchanged.

   \ingroup signal_func_fft_r2c
*/
AFAPI af_err af_fft_r2c (af_array *out, const af_array in, const double norm_factor, const dim_t pad0);
#endif

#if AF_API_VERSION >= 31
/**
   C Interface for real to complex fast fourier transform for two dimensional signals

   \param[out] out is a complex array containing the non redundant parts of \p in.
   \param[in]  in is a real array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  pad0 is the length of output signals along first dimension - used to either truncate/pad the input
   \param[in]  pad1 is the length of output signals along second dimension - used to either truncate/pad the input
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \note The first dimension of the output will be of size (pad0 / 2) + 1. The second dimension of the output will be pad1. The remaining dimensions are unchanged.

   \ingroup signal_func_fft_r2c
*/
AFAPI af_err af_fft2_r2c(af_array *out, const af_array in, const double norm_factor, const dim_t pad0, const dim_t pad1);
#endif

#if AF_API_VERSION >= 31
/**
   C Interface for real to complex fast fourier transform for three dimensional signals

   \param[out] out is a complex array containing the non redundant parts of \p in.
   \param[in]  in is a real array
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  pad0 is the length of output signals along first dimension - used to either truncate/pad the input
   \param[in]  pad1 is the length of output signals along second dimension - used to either truncate/pad the input
   \param[in]  pad2 is the length of output signals along third dimension - used to either truncate/pad the input
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \note The first dimension of the output will be of size (pad0 / 2) + 1. The second dimension of the output will be pad1. The third dimension of the output will be pad 2.

   \ingroup signal_func_fft_r2c
*/
AFAPI af_err af_fft3_r2c(af_array *out, const af_array in, const double norm_factor, const dim_t pad0, const dim_t pad1, const dim_t pad2);
#endif

#if AF_API_VERSION >= 31
/**
   C Interface for complex to real fast fourier transform for one dimensional signals

   \param[out] out is a real array containing the output of the transform.
   \param[in]  in is a complex array containing only the non redundant parts of the signals.
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  is_odd is a flag signifying if the output should be even or odd size
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \note The first dimension of the output will be 2 * dim0 - 1 if is_odd is true else 2 * dim0 - 2 where dim0 is the first dimension of the input. The remaining dimensions are unchanged.

   \ingroup signal_func_fft_c2r
*/

AFAPI af_err af_fft_c2r (af_array *out, const af_array in, const double norm_factor, const bool is_odd);
#endif

#if AF_API_VERSION >= 31
/**
   C Interface for complex to real fast fourier transform for two dimensional signals

   \param[out] out is a real array containing the output of the transform.
   \param[in]  in is a complex array containing only the non redundant parts of the signals.
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  is_odd is a flag signifying if the output should be even or odd size
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \note The first dimension of the output will be 2 * dim0 - 1 if is_odd is true else 2 * dim0 - 2 where dim0 is the first dimension of the input. The remaining dimensions are unchanged.

   \ingroup signal_func_fft_c2r
*/
AFAPI af_err af_fft2_c2r(af_array *out, const af_array in, const double norm_factor, const bool is_odd);
#endif

#if AF_API_VERSION >= 31
/**
   C Interface for complex to real fast fourier transform for three dimensional signals

   \param[out] out is a real array containing the output of the transform.
   \param[in]  in is a complex array containing only the non redundant parts of the signals.
   \param[in]  norm_factor is the normalization factor with which the input is scaled before the transformation is applied
   \param[in]  is_odd is a flag signifying if the output should be even or odd size
   \return     \ref AF_SUCCESS if the fft transform is successful,
               otherwise an appropriate error code is returned.

   \note The first dimension of the output will be 2 * dim0 - 1 if is_odd is true else 2 * dim0 - 2 where dim0 is the first dimension of the input. The remaining dimensions are unchanged.

   \ingroup signal_func_fft_c2r
*/
AFAPI af_err af_fft3_c2r(af_array *out, const af_array in, const double norm_factor, const bool is_odd);
#endif

/**
   C Interface for convolution on one dimensional signals

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
   C Interface for convolution on two dimensional signals

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
   C Interface for convolution on three dimensional signals

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
   C Interface for separable convolution on two dimensional signals

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
   C Interface for FFT-based convolution on one dimensional signals

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
   C Interface for FFT-based convolution on two dimensional signals

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
   C Interface for FFT-based convolution on three dimensional signals

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
