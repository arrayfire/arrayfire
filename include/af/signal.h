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
   C++ Interface for data interpolation on one-dimensional signals.

   \param[in]  in is the multidimensional input array. Values assumed to lie uniformly spaced indices in the range of `[0, n)`, where `n` is the number of elements in the array.
   \param[in]  pos positions of the interpolation points along the first dimension.
   \param[in]  method is the interpolation method to be used. The following types (defined in enum \ref af_interp_type) are supported: nearest neighbor, linear, and cubic.
   \param[in]  off_grid is the default value for any indices outside the valid range of indices.
   \returns    the interpolated array.

   The code sample below demonstrates approx1()'s usage:

   \snippet test/approx1.cpp ex_signal_approx1

   \ingroup signal_func_approx1
 */
AFAPI array approx1(const array &in, const array &pos,
                    const interpType method = AF_INTERP_LINEAR, const float off_grid = 0.0f);

/**
   C++ Interface for data interpolation on two-dimensional signals.

   \param[in]  in is the multidimensional input array. Values assumed to lie uniformly spaced indices in the range of `[0, n)` along both interpolation dimensions. `n` is the number of elements in the array.
   \param[in]  pos0 positions of the interpolation points along the first dimension.
   \param[in]  pos1 positions of the interpolation points along the second dimension.
   \param[in]  method is the interpolation method to be used. All interpolation types defined in \ref af_interp_type are supported.
   \param[in]  off_grid is the default value for any indices outside the valid range of indices.
   \returns    the interpolated array.

   The code sample below demonstrates approx2()'s usage:

   \snippet test/approx2.cpp ex_signal_approx2

   \ingroup signal_func_approx2
 */
AFAPI array approx2(const array &in, const array &pos0, const array &pos1,
                    const interpType method = AF_INTERP_LINEAR, const float off_grid = 0.0f);


#if AF_API_VERSION >= 37
/**
   C++ Interface for data interpolation on one-dimensional signals.

   The following version of approx1() accepts the dimension to perform
   the interpolation along the input. It also accepts start and step
   values which define the uniform range of corresponding indices.

   The following image illustrates what the range of indices
   corresponding to the input values look like if `idx_start` and
   `idx_step` are set to an arbitrary value of 10,

   \image html approx1_arbitrary_idx.png "approx1() using idx_start=10.0, idx_step=10.0"

   The blue dots represent indices whose values are known. The red dots
   represent indices whose values are unknown.

   \param[in]  in is the multidimensional input array. Values lie on uniformly spaced indices determined by `idx_start` and `idx_step`.
   \param[in]  pos positions of the interpolation points along `interp_dim`.
   \param[in]  interp_dim is the dimension to perform interpolation across.
   \param[in]  idx_start is the first index value along `interp_dim`.
   \param[in]  idx_step is the uniform spacing value between subsequent indices along `interp_dim`.
   \param[in]  method is the interpolation method to be used. The following types (defined in enum \ref af_interp_type) are supported: nearest neighbor, linear, and cubic.
   \param[in]  off_grid is the default value for any indices outside the valid range of indices.
   \returns    the interpolated array.

   The code sample below demonstrates usage:

   \snippet test/approx1.cpp ex_signal_approx1_uniform

   \ingroup signal_func_approx1
 */
AFAPI array approx1(const array &in,
                    const array &pos, const int interp_dim,
                    const double idx_start, const double idx_step,
                    const interpType method = AF_INTERP_LINEAR, const float off_grid = 0.0f);

/**
   C++ Interface for data interpolation on two-dimensional signals.

   The following version of the approx2() accepts the two dimensions
   to perform the interpolation along the input. It also accepts start
   and step values which define the uniform range of corresponding
   indices.

   \param[in]  in is the multidimensional input array.
   \param[in]  pos0 positions of the interpolation points along `interp_dim0`.
   \param[in]  interp_dim0 is the first dimension to perform interpolation across.
   \param[in]  idx_start_dim0 is the first index value along `interp_dim0`.
   \param[in]  idx_step_dim0 is the uniform spacing value between subsequent indices along `interp_dim0`.
   \param[in]  pos1 positions of the interpolation points along `interp_dim1`.
   \param[in]  interp_dim1 is the second dimension to perform interpolation across.
   \param[in]  idx_start_dim1 is the first index value along `interp_dim1`.
   \param[in]  idx_step_dim1 is the uniform spacing value between subsequent indices along `interp_dim1`.
   \param[in]  method is the interpolation method to be used. All interpolation types defined in \ref af_interp_type are supported.
   \param[in]  off_grid is the default value for any indices outside the valid range of indices.
   \returns    the interpolated array.

   The code sample below demonstrates usage:

   \snippet test/approx2.cpp ex_signal_approx2_uniform

   \ingroup signal_func_approx2
 */
AFAPI array approx2(const array &in,
                    const array &pos0, const int interp_dim0, const double idx_start_dim0, const double idx_step_dim0,
                    const array &pos1, const int interp_dim1, const double idx_start_dim1, const double idx_step_dim1,
                    const interpType method = AF_INTERP_LINEAR, const float off_grid = 0.0f);
#endif

/**
   C++ Interface for fast fourier transform on one dimensional signals

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
   \param[in]  odim0 is the length of output signals - used to either truncate or pad the input signals
   \return     the transformed array

   \ingroup signal_func_fft
 */
AFAPI array fftNorm(const array& in, const double norm_factor, const dim_t odim0=0);

/**
   C++ Interface for fast fourier transform on two dimensional signals

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
   \param[in]  odim0 is the length of output signals along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output signals along second dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_fft2
 */
AFAPI array fft2Norm(const array& in, const double norm_factor, const dim_t odim0=0, const dim_t odim1=0);

/**
   C++ Interface for fast fourier transform on three dimensional signals

   \param[in]  in is the input array and the output of 1D fourier transform on exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied

   \note The input \p in must be complex

   \ingroup signal_func_fft
 */
AFAPI void fftInPlace(array& in, const double norm_factor = 1.0);
#endif

#if AF_API_VERSION >= 31
/**
   C++ Interface for fast fourier transform on two dimensional signals

   \param[inout]  in is the input array on entry and the output of 2D forward fourier transform on exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied

   \note The input \p in must be complex

   \ingroup signal_func_fft2
 */
AFAPI void fft2InPlace(array& in, const double norm_factor = 1.0);
#endif

#if AF_API_VERSION >= 31
/**
   C++ Interface for fast fourier transform on three dimensional signals

   \param[inout]  in is the input array on entry and the output of 3D forward fourier transform on exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied

   \note The input \p in must be complex

   \ingroup signal_func_fft3
 */
AFAPI void fft3InPlace(array& in, const double norm_factor = 1.0);
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
   \param[in]  odim0 is the length of output signals - used to either truncate or pad the input signals
   \return     the transformed array

   \ingroup signal_func_ifft
 */
AFAPI array ifftNorm(const array& in, const double norm_factor, const dim_t odim0=0);

/**
   C++ Interface for inverse fast fourier transform on two dimensional signals

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
   \param[in]  odim0 is the length of output signals along first dimension - used to either truncate/pad the input
   \param[in]  odim1 is the length of output signals along second dimension - used to either truncate/pad the input
   \return     the transformed array

   \ingroup signal_func_ifft2
 */
AFAPI array ifft2Norm(const array& in, const double norm_factor, const dim_t odim0=0, const dim_t odim1=0);

/**
   C++ Interface for inverse fast fourier transform on three dimensional signals

   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied

   \note The input \p in must be complex

   \ingroup signal_func_ifft
 */
AFAPI void ifftInPlace(array& in, const double norm_factor = 1.0);
#endif

#if AF_API_VERSION >= 31
/**
   C++ Interface for fast fourier transform on two dimensional signals

   \param[inout]  in is the input array on entry and the output of 2D inverse fourier transform on exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied

   \note The input \p in must be complex

   \ingroup signal_func_ifft2
 */
AFAPI void ifft2InPlace(array& in, const double norm_factor = 1.0);
#endif

#if AF_API_VERSION >= 31
/**
   C++ Interface for fast fourier transform on three dimensional signals

   \param[inout]  in is the input array on entry and the output of 3D inverse fourier transform on exit
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied

   \note The input \p in must be complex

   \ingroup signal_func_ifft3
 */
AFAPI void ifft3InPlace(array& in, const double norm_factor = 1.0);
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
   \return     a complex array containing the non redundant parts of \p in along the first dimension.

   \note The first dimension of the output will be of size (dims[0] / 2) + 1. The remaining dimensions are unchanged.

   \ingroup signal_func_fft_r2c
*/
template<int rank>
array fftR2C(const array &in,
             const dim4& dims,
             const double norm_factor = 1.0);
#endif

#if AF_API_VERSION >= 31
/**
   C++ Interface for real to complex fast fourier transform for one dimensional signals

   \param[in]  in is a real array
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
   \return     a complex array containing the non redundant parts of \p in along the first dimension.

   \note The first dimension of the output will be of size (in.dims(0) / 2) + 1. The remaining dimensions are unchanged.

   \ingroup signal_func_fft_r2c
*/
template<int rank>
array fftR2C(const array &in,
             const double norm_factor = 1.0);
#endif

#if AF_API_VERSION >= 31
/**
   C++ Interface for complex to real fast fourier transform

   \param[in]  in is a complex array containing only the non redundant parts of the signals
   \param[in]  is_odd is a flag signifying if the output should be even or odd size
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
   \tparam     rank signifies the dimensionality of the transform
   \return     A real array of size [2 * idim0 - 2 + is_odd, idim1, idim2, idim3] where idim{0,1,2,3} signify input dimensions

   \ingroup signal_func_fft_c2r
*/

template<int rank>
array fftC2R(const array &in, bool is_odd = false,
                 const double norm_factor = 1.0);
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

   \ingroup signal_func_convolve_sep
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
   C++ Interface for 2D convolution

   This version of convolution is consistent with the machine learning
   formulation that will spatially convolve a filter on 2-dimensions against a
   signal. Multiple signals and filters can be batched against each other.
   Furthermore, the signals and filters can be multi-dimensional however their
   dimensions must match.

   Example:
   Signals with dimensions: d0 x d1 x d2 x Ns
   Filters with dimensions: d0 x d1 x d2 x Nf

   Resulting Convolution: d0 x d1 x Nf x Ns

   \param[in]  signal   is the input signal
   \param[in]  filter   is the filter that will be used for the convolution operation
   \param[in]  stride   specifies the filter strides along each dimension
   \param[in]  padding  specifies the padding along each dimension
   \param[in]  dilation specifies the amount to dilate the filter before convolution
   \return              the convolved array

   \note Make sure you pass in both dim0, and dim1 in your dim4 arguments. The third
   and fourth dimensions are currently ignored.

   \ingroup signal_func_convolve2
 */
AFAPI array convolve2NN(const array& signal, const array& filter,
                        const dim4 stride, const dim4 padding, const dim4 dilation);

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

   \ingroup signal_func_convolve
 */
AFAPI array fftConvolve(const array& signal, const array& filter, const convMode mode=AF_CONV_DEFAULT);

/**
   C++ Interface for convolution on 1D signals using FFT

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     the convolved array

   \ingroup signal_func_convolve1
 */
AFAPI array fftConvolve1(const array& signal, const array& filter, const convMode mode=AF_CONV_DEFAULT);

/**
   C++ Interface for convolution on 2D signals using FFT

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     the convolved array

   \ingroup signal_func_convolve2
 */
AFAPI array fftConvolve2(const array& signal, const array& filter, const convMode mode=AF_CONV_DEFAULT);

/**
   C++ Interface for convolution on 3D signals using FFT

   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     the convolved array

   \ingroup signal_func_convolve3
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

/**
    C++ Interface for median filter

    \snippet test/medfilt.cpp ex_image_medfilt

    \param[in]  in array is the input image
    \param[in]  wind_length is the kernel height
    \param[in]  wind_width is the kernel width
    \param[in]  edge_pad value will decide what happens to border when running
                filter in their neighborhood. It takes one of the values [\ref AF_PAD_ZERO | \ref AF_PAD_SYM]
    \return     the processed image

    \ingroup image_func_medfilt
*/
AFAPI array medfilt(const array& in, const dim_t wind_length = 3, const dim_t wind_width = 3, const borderType edge_pad = AF_PAD_ZERO);

#if AF_API_VERSION >= 34
/**
    C++ Interface for median filter

    \snippet test/medfilt.cpp ex_image_medfilt

    \param[in]  in array is the input signal
    \param[in]  wind_width is the kernel width
    \param[in]  edge_pad value will decide what happens to border when running
                filter in their neighborhood. It takes one of the values [\ref AF_PAD_ZERO | \ref AF_PAD_SYM]
    \return     the processed signal

    \ingroup image_func_medfilt
*/
AFAPI array medfilt1(const array& in, const dim_t wind_width = 3, const borderType edge_pad = AF_PAD_ZERO);
#endif

#if AF_API_VERSION >= 34
/**
    C++ Interface for median filter

    \snippet test/medfilt.cpp ex_image_medfilt

    \param[in]  in array is the input image
    \param[in]  wind_length is the kernel height
    \param[in]  wind_width is the kernel width
    \param[in]  edge_pad value will decide what happens to border when running
                filter in their neighborhood. It takes one of the values [\ref AF_PAD_ZERO | \ref AF_PAD_SYM]
    \return     the processed image

    \ingroup image_func_medfilt
*/
AFAPI array medfilt2(const array& in, const dim_t wind_length = 3, const dim_t wind_width = 3, const borderType edge_pad = AF_PAD_ZERO);
#endif

#if AF_API_VERSION >= 35
/**
   C++ Interface for setting plan cache size

   This function doesn't do anything if called when CPU backend is active. The plans associated with
   the most recently used array sizes are cached.

   \param[in] cacheSize is the number of plans that shall be cached
*/
AFAPI void setFFTPlanCacheSize(size_t cacheSize);
#endif

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
   C Interface for signals interpolation on one dimensional signals.

   \param[out] out      is the interpolated array.
   \param[in]  in       is the multidimensional input array. Values assumed
                        to lie uniformly spaced indices in the range of
                        `[0, n)`, where `n` is the number of elements in the
                        array.
   \param[in]  pos      positions of the interpolation points along the first
                        dimension.
   \param[in]  method   is the interpolation method to be used. The following
                        types (defined in enum \ref af_interp_type)
                        are supported: nearest neighbor, linear, and cubic.
   \param[in]  off_grid is the default value for any indices outside the
                           valid range of indices.

   \return \ref AF_SUCCESS if the interpolation operation is successful,
           otherwise an appropriate error code is returned.

   \ingroup signal_func_approx1
 */
AFAPI af_err af_approx1(af_array *out, const af_array in, const af_array pos,
                        const af_interp_type method, const float off_grid);

#if AF_API_VERSION >= 37
/**
   C Interface for the version of \ref af_approx1 that accepts a preallocated
   output array

   \param[in,out] out      is the interpolated array (can be preallocated).
   \param[in]     in       is the multidimensional input array. Values assumed
                           to lie uniformly spaced indices in the range of
                           `[0, n)`, where `n` is the number of elements in the
                           array.
   \param[in]     pos      positions of the interpolation points along the first
                           dimension.
   \param[in]     method   is the interpolation method to be used. The following
                           types (defined in enum \ref af_interp_type)
                           are supported: nearest neighbor, linear, and cubic.
   \param[in]     off_grid is the default value for any indices outside the
                           valid range of indices.

   \return \ref AF_SUCCESS if the interpolation operation is successful,
           otherwise an appropriate error code is returned.

   \note \p out can either be a null or existing `af_array` object. If it is a
         sub-array of an existing `af_array`, only the corresponding portion of
         the `af_array` will be overwritten
   \note Passing an `af_array` that has not been initialized to \p out will
         cause undefined behavior.

   \ingroup signal_func_approx1
 */
AFAPI af_err af_approx1_v2(af_array *out, const af_array in, const af_array pos,
                           const af_interp_type method, const float off_grid);
#endif

/**
   C Interface for signals interpolation on two dimensional signals.

   \param[out] out      the interpolated array.
   \param[in]  in       is the multidimensional input array. Values assumed to
                        lie uniformly spaced indices in the range of `[0, n)`
                        along both interpolation dimensions. `n` is the number
                        of elements in the array.
   \param[in]  pos0     positions of the interpolation points along the first
                        dimension.
   \param[in]  pos1     positions of the interpolation points along the second
                        dimension.
   \param[in]  method   is the interpolation method to be used. All
                        interpolation types defined in \ref af_interp_type are
                        supported.
   \param[in]  off_grid is the default value for any indices outside the valid
                        range of indices.

   \return \ref AF_SUCCESS if the interpolation operation is successful,
           otherwise an appropriate error code is returned.

   \ingroup signal_func_approx2
 */
AFAPI af_err af_approx2(af_array *out, const af_array in,
                        const af_array pos0, const af_array pos1,
                        const af_interp_type method, const float off_grid);

#if AF_API_VERSION >= 37
/**
   C Interface for the version of \ref af_approx2 that accepts a preallocated
   output array

   \param[in,out] out      the interpolated array (can be preallocated).
   \param[in]     in       is the multidimensional input array. Values assumed
                           to lie uniformly spaced indices in the range of
                           `[0, n)` along both interpolation dimensions. `n` is
                           the number of elements in the array.
   \param[in]     pos0     positions of the interpolation points along the first
                           dimension.
   \param[in]     pos1     positions of the interpolation points along the
                           second dimension.
   \param[in]     method   is the interpolation method to be used. All
                           interpolation types defined in \ref af_interp_type
                           are supported.
   \param[in]     off_grid is the default value for any indices outside the
                           valid range of indices.

   \return \ref AF_SUCCESS if the interpolation operation is successful,
           otherwise an appropriate error code is returned.

   \note \p out can either be a null or existing `af_array` object. If it is a
         sub-array of an existing `af_array`, only the corresponding portion of
         the `af_array` will be overwritten
   \note Passing an `af_array` to \p out that has not been initialized will
         cause undefined behavior.

   \ingroup signal_func_approx2
 */
AFAPI af_err af_approx2_v2(af_array *out, const af_array in,
                           const af_array pos0, const af_array pos1,
                           const af_interp_type method, const float off_grid);
#endif


#if AF_API_VERSION >= 37
/**
   C Interface for signals interpolation on one dimensional signals along
   specified dimension.

   af_approx1_uniform() accepts the dimension to perform the interpolation along
   the input. It also accepts start and step values which define the uniform
   range of corresponding indices.

   The following image illustrates what the range of indices corresponding to
   the input values look like if `idx_start` and `idx_step` are set to an
   arbitrary value of 10,

   \image html approx1_arbitrary_idx.png "approx1() using idx_start=10.0, idx_step=10.0"

   The blue dots represent indices whose values are known. The red dots
   represent indices whose values are unknown.

   \param[out] out        the interpolated array.
   \param[in]  in         is the multidimensional input array. Values lie on
                          uniformly spaced indices determined by `idx_start`
                          and `idx_step`.
   \param[in]  pos        positions of the interpolation points along
                          `interp_dim`.
   \param[in]  interp_dim is the dimension to perform interpolation across.
   \param[in]  idx_start  is the first index value along `interp_dim`.
   \param[in]  idx_step   is the uniform spacing value between subsequent
                          indices along `interp_dim`.
   \param[in]  method     is the interpolation method to be used. The
                          following types (defined in enum
                          \ref af_interp_type) are supported: nearest
                          neighbor, linear, and cubic.
   \param[in]  off_grid   is the default value for any indices outside the
                          valid range of indices.

   \return \ref AF_SUCCESS if the interpolation operation is successful,
           otherwise an appropriate error code is returned.

   \ingroup signal_func_approx1
 */
AFAPI af_err af_approx1_uniform(af_array *out, const af_array in,
                                const af_array pos, const int interp_dim,
                                const double idx_start, const double idx_step,
                                const af_interp_type method,
                                const float off_grid);

/**
   C Interface for the version of \ref af_approx1_uniform that accepts a
   preallocated output array

   \param[in,out] out        the interpolated array (can be preallocated).
   \param[in]     in         is the multidimensional input array. Values lie on
                             uniformly spaced indices determined by `idx_start`
                             and `idx_step`.
   \param[in]     pos        positions of the interpolation points along
                             `interp_dim`.
   \param[in]     interp_dim is the dimension to perform interpolation across.
   \param[in]     idx_start  is the first index value along `interp_dim`.
   \param[in]     idx_step   is the uniform spacing value between subsequent
                             indices along `interp_dim`.
   \param[in]     method     is the interpolation method to be used. The
                             following types (defined in enum
                             \ref af_interp_type) are supported: nearest
                             neighbor, linear, and cubic.
   \param[in]     off_grid   is the default value for any indices outside the
                             valid range of indices.

   \return \ref AF_SUCCESS if the interpolation operation is successful,
           otherwise an appropriate error code is returned.

   \note \p out can either be a null or existing `af_array` object. If it is a
         sub-array of an existing `af_array`, only the corresponding portion of
         the `af_array` will be overwritten
   \note Passing an `af_array` to \p out that has not been initialized will
         cause undefined behavior.

   \ingroup signal_func_approx1
 */
AFAPI af_err af_approx1_uniform_v2(af_array *out, const af_array in,
                                   const af_array pos, const int interp_dim,
                                   const double idx_start,
                                   const double idx_step,
                                   const af_interp_type method,
                                   const float off_grid);

/**
   C Interface for signals interpolation on two dimensional signals along
   specified dimensions.

   af_approx2_uniform() accepts two dimensions to perform the interpolation
   along the input. It also accepts start and step values which define the
   uniform range of corresponding indices.

   \param[out] out            the interpolated array.
   \param[in]  in             is the multidimensional input array.
   \param[in]  pos0           positions of the interpolation points along
                              `interp_dim0`.
   \param[in]  interp_dim0    is the first dimension to perform interpolation
                              across.
   \param[in]  idx_start_dim0 is the first index value along `interp_dim0`.
   \param[in]  idx_step_dim0  is the uniform spacing value between subsequent
                              indices along `interp_dim0`.
   \param[in]  pos1           positions of the interpolation points along
                              `interp_dim1`.
   \param[in]  interp_dim1    is the second dimension to perform interpolation
                              across.
   \param[in]  idx_start_dim1 is the first index value along `interp_dim1`.
   \param[in]  idx_step_dim1  is the uniform spacing value between subsequent
                              indices along `interp_dim1`.
   \param[in]  method         is the interpolation method to be used. All
                              interpolation types defined in \ref af_interp_type
                              are supported.
   \param[in]  off_grid       is the default value for any indices outside the
                              valid range of indices.

   \return \ref AF_SUCCESS if the interpolation operation is successful,
           otherwise an appropriate error code is returned.

   \ingroup signal_func_approx2
 */
AFAPI af_err af_approx2_uniform(af_array *out, const af_array in,
                                const af_array pos0, const int interp_dim0,
                                const double idx_start_dim0,
                                const double idx_step_dim0,
                                const af_array pos1, const int interp_dim1,
                                const double idx_start_dim1,
                                const double idx_step_dim1,
                                const af_interp_type method,
                                const float off_grid);

/**
   C Interface for the version of \ref af_approx2_uniform that accepts a
   preallocated output array

   \param[in,out] out            the interpolated array.
   \param[in]     in             is the multidimensional input array.
   \param[in]     pos0           positions of the interpolation points along
                                 `interp_dim0`.
   \param[in]     interp_dim0    is the first dimension to perform interpolation
                                 across.
   \param[in]     idx_start_dim0 is the first index value along `interp_dim0`.
   \param[in]     idx_step_dim0  is the uniform spacing value between subsequent
                                 indices along `interp_dim0`.
   \param[in]     pos1           positions of the interpolation points along
                                 `interp_dim1`.
   \param[in]     interp_dim1    is the second dimension to perform
                                 interpolation across.
   \param[in]     idx_start_dim1 is the first index value along `interp_dim1`.
   \param[in]     idx_step_dim1  is the uniform spacing value between subsequent
                                 indices along `interp_dim1`.
   \param[in]     method         is the interpolation method to be used. All
                                 interpolation types defined in
                                 \ref af_interp_type are supported.
   \param[in]     off_grid       is the default value for any indices outside
                                 the valid range of indices.

   \return \ref AF_SUCCESS if the interpolation operation is successful,
           otherwise an appropriate error code is returned.

   \note \p out can either be a null or existing `af_array` object. If it is a
         sub-array of an existing `af_array`, only the corresponding portion of
         the `af_array` will be overwritten
   \note Passing an `af_array` to \p out that has not been initialized will
         cause undefined behavior.

   \ingroup signal_func_approx2
 */
AFAPI af_err af_approx2_uniform_v2(af_array *out, const af_array in,
                                   const af_array pos0, const int interp_dim0,
                                   const double idx_start_dim0,
                                   const double idx_step_dim0,
                                   const af_array pos1, const int interp_dim1,
                                   const double idx_start_dim1,
                                   const double idx_step_dim1,
                                   const af_interp_type method,
                                   const float off_grid);
#endif

/**
   C Interface for fast fourier transform on one dimensional signals

   \param[out] out is the transformed array
   \param[in]  in is the input array
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   \param[in]  norm_factor is the normalization factor with which the input is scaled after the transformation is applied
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
   C Interface for 2D convolution

   This version of convolution is consistent with the machine learning
   formulation that will spatially convolve a filter on 2-dimensions against a
   signal. Multiple signals and filters can be batched against each other.
   Furthermore, the signals and filters can be multi-dimensional however their
   dimensions must match.

   Example:
   Signals with dimensions: d0 x d1 x d2 x Ns
   Filters with dimensions: d0 x d1 x d2 x Nf

   Resulting Convolution: d0 x d1 x Nf x Ns

   \param[out] out is convolved array
   \param[in]  signal is the input signal
   \param[in]  filter is the filter that will be used for the convolution operation
   \param[in]  stride_dims specifies the number of stride dimension parameters
   \param[in]  strides array of values specifying the amounts the filter strides along each dimension
   \param[in]  padding_dims specifies the number of padding dimension parameters
   \param[in]  paddings array of values specifying the amounts to pad along each dimension
   \param[in]  dilation_dims specifies the number of dilation dimension parameters
   \param[in]  dilations array of values specifying the amounts to dilate the filter
               before convolving along each dimension
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_convolve2
 */
AFAPI af_err af_convolve2_nn(af_array *out, const af_array signal, const af_array filter,
                             const unsigned stride_dims,   const dim_t *strides,
                             const unsigned padding_dims,  const dim_t *paddings,
                             const unsigned dilation_dims, const dim_t *dilations);

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

   \ingroup signal_func_convolve_sep
 */
AFAPI af_err af_convolve2_sep(af_array *out, const af_array col_filter, const af_array row_filter, const af_array signal, const af_conv_mode mode);

/**
   C Interface for convolution on 1D signals using FFT

   \param[out] out is convolved array
   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_convolve1
 */
AFAPI af_err af_fft_convolve1(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode);

/**
   C Interface for convolution on 2D signals using FFT

   \param[out] out is convolved array
   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_convolve2
 */
AFAPI af_err af_fft_convolve2(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode);

/**
   C Interface for convolution on 3D signals using FFT

   \param[out] out is convolved array
   \param[in]  signal is the input signal
   \param[in]  filter is the signal that shall be used for the convolution operation
   \param[in]  mode indicates if the convolution should be expanded or not(where output size equals input)
   \return     \ref AF_SUCCESS if the convolution is successful,
               otherwise an appropriate error code is returned.

   \ingroup signal_func_convolve3
 */
AFAPI af_err af_fft_convolve3(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode);

/**
   C Interface for finite impulse response  filter

   \param[out] y is the output signal from the filter
   \param[in] b is the array containing the coefficients of the filter
   \param[in] x is the input signal to the filter

   \ingroup signal_func_fir
*/
AFAPI af_err af_fir(af_array *y, const af_array b, const af_array x);

/**
   C Interface for infinite impulse response filter

   \param[out] y is the output signal from the filter
   \param[in] b is the array containing the feedforward coefficients
   \param[in] a is the array containing the feedback coefficients
   \param[in] x is the input signal to the filter

   \note The feedforward coefficients are currently limited to a length of 512

   \ingroup signal_func_iir
*/
AFAPI af_err af_iir(af_array *y, const af_array b, const af_array a, const af_array x);

    /**
        C Interface for median filter

        \param[out] out array is the processed image
        \param[in]  in array is the input image
        \param[in]  wind_length is the kernel height
        \param[in]  wind_width is the kernel width
        \param[in]  edge_pad value will decide what happens to border when running
                    filter in their neighborhood. It takes one of the values [\ref AF_PAD_ZERO | \ref AF_PAD_SYM]
        \return     \ref AF_SUCCESS if the median filter is applied successfully,
        otherwise an appropriate error code is returned.

        \ingroup image_func_medfilt
    */
    AFAPI af_err af_medfilt(af_array *out, const af_array in, const dim_t wind_length, const dim_t wind_width, const af_border_type edge_pad);

#if AF_API_VERSION >= 34
    /**
        C Interface for 1D median filter

        \param[out] out array is the processed signal
        \param[in]  in array is the input signal
        \param[in]  wind_width is the kernel width
        \param[in]  edge_pad value will decide what happens to border when running
                    filter in their neighborhood. It takes one of the values [\ref AF_PAD_ZERO | \ref AF_PAD_SYM]
        \return     \ref AF_SUCCESS if the median filter is applied successfully,
        otherwise an appropriate error code is returned.

        \ingroup image_func_medfilt
    */
    AFAPI af_err af_medfilt1(af_array *out, const af_array in, const dim_t wind_width, const af_border_type edge_pad);
#endif

#if AF_API_VERSION >= 34
    /**
        C Interface for median filter

        \param[out] out array is the processed image
        \param[in]  in array is the input image
        \param[in]  wind_length is the kernel height
        \param[in]  wind_width is the kernel width
        \param[in]  edge_pad value will decide what happens to border when running
                    filter in their neighborhood. It takes one of the values [\ref AF_PAD_ZERO | \ref AF_PAD_SYM]
        \return     \ref AF_SUCCESS if the median filter is applied successfully,
        otherwise an appropriate error code is returned.

        \ingroup image_func_medfilt
    */
    AFAPI af_err af_medfilt2(af_array *out, const af_array in, const dim_t wind_length, const dim_t wind_width, const af_border_type edge_pad);
#endif


#if AF_API_VERSION >= 34
/**
   C Interface for setting plan cache size

   This function doesn't do anything if called when CPU backend is active. The plans associated with
   the most recently used array sizes are cached.

   \param[in] cache_size is the number of plans that shall be cached

   \ingroup signal_func_fft
*/
AFAPI af_err af_set_fft_plan_cache_size(size_t cache_size);
#endif

#ifdef __cplusplus
}
#endif
