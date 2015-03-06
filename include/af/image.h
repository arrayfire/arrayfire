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
#include <af/features.h>

#ifdef __cplusplus
#include <utility>
namespace af
{

AFAPI array loadImage(const char* filename, const bool is_color=false);

AFAPI void saveImage(const char* filename, const array& in);

AFAPI array resize(const array& in, const dim_type odim0, const dim_type odim1, const interpType method=AF_INTERP_NEAREST);

AFAPI array resize(const float scale0, const float scale1, const array& in, const interpType method=AF_INTERP_NEAREST);

AFAPI array resize(const float scale, const array& in, const interpType method=AF_INTERP_NEAREST);

AFAPI array rotate(const array& in, const float theta, const bool crop=true, const interpType method=AF_INTERP_NEAREST);

AFAPI array transform(const array& in, const array& transform, const dim_type odim0, const dim_type odim1, const interpType method=AF_INTERP_NEAREST, const bool inverse=true);

AFAPI array translate(const array& in, const float trans0, const float trans1, const dim_type odim0, const dim_type odim1, const interpType method=AF_INTERP_NEAREST);

AFAPI array scale(const array& in, const float scale0, const float scale1, const dim_type odim0, const dim_type odim1, const interpType method=AF_INTERP_NEAREST);

AFAPI array skew(const array& in, const float skew0, const float skew1, const dim_type odim0, const dim_type odim1, const bool inverse=true, const interpType method=AF_INTERP_NEAREST);

AFAPI array bilateral(const array &in, const float spatial_sigma, const float chromatic_sigma, bool is_color=false);

/**
   C++ Interface

   \snippet test/histogram.cpp ex_image_hist_minmax

   \param[in]  in is the input array
   \param[in]  nbins  Number of bins to populate between min and max
   \param[in]  minval minimum bin value (accumulates -inf to min)
   \param[in]  maxval minimum bin value (accumulates max to +inf)
   \return     histogram array

   \ingroup image_func_histogram
 */
AFAPI array histogram(const array &in, const unsigned nbins, const double minval, const double maxval);

/**
   C++ Interface

   \snippet test/histogram.cpp ex_image_hist_nominmax

   \param[in]  in is the input array
   \param[in]  nbins  Number of bins to populate between min and max
   \return     histogram array

   \ingroup image_func_histogram
 */
AFAPI array histogram(const array &in, const unsigned nbins);

AFAPI array meanshift(const array& in, const float spatial_sigma, const float chromatic_sigma, const unsigned iter, const bool is_color=false);

AFAPI array medfilt(const array& in, dim_type wind_length = 3, dim_type wind_width = 3, padType edge_pad = AF_ZERO);

AFAPI array dilate(const array& in, const array& mask);

AFAPI array dilate3d(const array& in, const array& mask);

AFAPI array erode(const array& in, const array& mask);

AFAPI array erode3d(const array& in, const array& mask);

AFAPI void grad(array& rows, array& cols, const array& in);

AFAPI array regions(const array& in, af::connectivity connectivity=AF_CONNECTIVITY_4, dtype type=f32);

AFAPI features fast(const array& in, const float thr=20.0f, const unsigned arc_length=9, const bool non_max=true, const float feature_ratio=0.05, const unsigned edge=3);

AFAPI void orb(features& feat, array& desc, const array& image, const float fast_thr=20.f, const unsigned max_feat=400, const float scl_fctr=1.5f, const unsigned levels=4, const bool blur_img=false);

AFAPI array matchTemplate(const array &searchImg, const array &templateImg, matchType mType=AF_SAD);

AFAPI void sobel(array &dx, array &dy, const array &img, const unsigned ker_size=3);

AFAPI array sobel(const array &img, const unsigned ker_size=3, bool isFast=false);

/**
   C++ Interface

   \param[in]  in is an array in the RGB colorspace
   \param[in]  rPercent is percentage of red channel value contributing to grayscale intensity
   \param[in]  gPercent is percentage of green channel value contributing to grayscale intensity
   \param[in]  bPercent is percentage of blue channel value contributing to grayscale intensity
   \return     array in Grayscale colorspace

   \note \p in must be three dimensional for RGB to Grayscale conversion.

   \ingroup image_func_rgb2gray
 */
AFAPI array rgb2gray(const array& in, const float rPercent=0.2126f, const float gPercent=0.7152f, const float bPercent=0.0722f);

/**
   C++ Interface

   \param[in]  in is an array in the Grayscale colorspace
   \param[in]  rFactor is percentage of intensity value contributing to red channel
   \param[in]  gFactor is percentage of intensity value contributing to green channel
   \param[in]  bFactor is percentage of intensity value contributing to blue channel
   \return     array in RGB colorspace

   \note \p in must be two dimensional for Grayscale to RGB conversion.

   \ingroup image_func_gray2rgb
 */
AFAPI array gray2rgb(const array& in, const float rFactor=1.0, const float gFactor=1.0, const float bFactor=1.0);

/**
   C++ Interface

   \snippet test/histogram.cpp ex_image_histequal

   \param[in]  in is the input array, non-normalized input (!! assumes values [0-255] !!)
   \param[in]  hist target histogram to approximate in output (based on # of bins)
   \return     data with histogram approximately equal to histogram

   \note \p in must be two dimensional.

   \ingroup image_func_histequal
 */
AFAPI array histequal(const array& in, const array& hist);

AFAPI array gaussianKernel(const int rows, const int cols, const double sig_r = 0, const double sig_c = 0);

/**
   C++ Interface

   \param[in]  in is an array in the HSV colorspace
   \return     array in RGB colorspace

   \note \p in must be three dimensional

   \ingroup image_func_hsv2rgb
 */
AFAPI array hsv2rgb(const array& in);

/**
   C++ Interface

   \param[in]  in is an array in the RGB colorspace
   \return     array in HSV colorspace

   \note \p in must be three dimensional

   \ingroup image_func_rgb2hsv
 */
AFAPI array rgb2hsv(const array& in);

/**
   C++ Interface

   \param[in]  image is the input array
   \param[in]  to is the target array colorspace
   \param[in]  from is the input array colorspace
   \return     array in target colorspace

   \note  \p image must be 3 dimensional for \ref AF_HSV to \ref AF_RGB, \ref AF_RGB to
   \ref AF_HSV, & \ref AF_RGB to \ref AF_GRAY transformations. For \ref AF_GRAY to \ref AF_RGB
   transformation, 2D array is expected.

   \ingroup image_func_colorspace
 */
AFAPI array colorspace(const array& image, CSpace to, CSpace from);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    // Image IO: Load and Save Image functions
    AFAPI af_err af_load_image(af_array *out, const char* filename, const bool isColor);

    AFAPI af_err af_save_image(const char* filename, const af_array in);

    // Resize an image/matrix/array
    AFAPI af_err af_resize(af_array *out, const af_array in, const dim_type odim0, const dim_type odim1, const af_interp_type method);

    // Transform an image using a 3x2 transformation matrix.
    // If the transform matrix is a forward transformation matrix, then inverse is false.
    // If the transform martix is an inverse transformation matrix, then inverse is true;
    AFAPI af_err af_transform(af_array *out, const af_array in, const af_array transform,
                              const dim_type odim0, const dim_type odim1,
                              const af_interp_type method, const bool inverse);

    // Rotate
    AFAPI af_err af_rotate(af_array *out, const af_array in, const float theta,
                           const bool crop, const af_interp_type method);
    // Translate
    AFAPI af_err af_translate(af_array *out, const af_array in, const float trans0, const float trans1,
                              const dim_type odim0, const dim_type odim1, const af_interp_type method);
    // Scale
    AFAPI af_err af_scale(af_array *out, const af_array in, const float scale0, const float scale1,
                          const dim_type odim0, const dim_type odim1, const af_interp_type method);
    // Skew
    AFAPI af_err af_skew(af_array *out, const af_array in, const float skew0, const float skew1,
                         const dim_type odim0, const dim_type odim1, const af_interp_type method,
                         const bool inverse);

    /**
       C Interface

       \param[out] out is the histogram for input array in
       \param[in]  in is the input array
       \param[in]  nbins  Number of bins to populate between min and max
       \param[in]  minval minimum bin value (accumulates -inf to min)
       \param[in]  maxval minimum bin value (accumulates max to +inf)
       \return     \ref AF_SUCCESS if the histogram is successfully created,
       otherwise an appropriate error code is returned.

       \ingroup image_func_histogram
     */
    AFAPI af_err af_histogram(af_array *out, const af_array in, const unsigned nbins, const double minval, const double maxval);

    // image dilation operation
    AFAPI af_err af_dilate(af_array *out, const af_array in, const af_array mask);

    AFAPI af_err af_dilate3d(af_array *out, const af_array in, const af_array mask);

    // image erosion operation
    AFAPI af_err af_erode(af_array *out, const af_array in, const af_array mask);

    AFAPI af_err af_erode3d(af_array *out, const af_array in, const af_array mask);

    // image bilateral filter
    AFAPI af_err af_bilateral(af_array *out, const af_array in, const float spatial_sigma, const float chromatic_sigma, const bool isColor);

    // image meanshift filter
    AFAPI af_err af_meanshift(af_array *out, const af_array in, const float spatial_sigma, const float chromatic_sigma, const unsigned iter, const bool is_color);

    // gradient
    AFAPI af_err af_gradient(af_array *grad_rows, af_array *grad_cols, const af_array in);

    // image median filter
    AFAPI af_err af_medfilt(af_array *out, const af_array in, dim_type wind_length, dim_type wind_width, af_pad_type edge_pad);

    // Compute labels for connected regions from binary input arrays
    AFAPI af_err af_regions(af_array *out, const af_array in, af_connectivity connectivity, af_dtype ty);

    // Compute FAST corners from input image
    AFAPI af_err af_fast(af_features *out, const af_array in, const float thr, const unsigned arc_length, const bool non_max, const float feature_ratio, const unsigned edge);

    // Compute FAST corners and ORB descriptors from input image
    AFAPI af_err af_orb(af_features *feat, af_array *desc, const af_array in, const float fast_thr, const unsigned max_feat, const float scl_fctr, const unsigned levels, const bool blur_img);

    // object detection algorithm, matching pattern image to target image and giving disparity results
    AFAPI af_err af_match_template(af_array *out, const af_array search_img, const af_array template_img, af_match_type m_type);

    // sobel operator for images
    AFAPI af_err af_sobel_operator(af_array *dx, af_array *dy, const af_array img, const unsigned ker_size);

    /**
       C Interface

       \param[out] out is an array in target colorspace
       \param[in]  in is an array in the RGB colorspace
       \param[in]  rPercent is percentage of red channel value contributing to grayscale intensity
       \param[in]  gPercent is percentage of green channel value contributing to grayscale intensity
       \param[in]  bPercent is percentage of blue channel value contributing to grayscale intensity
       \return     \ref AF_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note \p in must be three dimensional for RGB to Grayscale conversion.

       \ingroup image_func_rgb2gray
     */
    AFAPI af_err af_rgb2gray(af_array* out, const af_array in, const float rPercent, const float gPercent, const float bPercent);

    /**
       C Interface

       \param[out] out is an array in target colorspace
       \param[in]  in is an array in the Grayscale colorspace
       \param[in]  rFactor is percentage of intensity value contributing to red channel
       \param[in]  gFactor is percentage of intensity value contributing to green channel
       \param[in]  bFactor is percentage of intensity value contributing to blue channel
       \return     \ref AF_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note \p in must be two dimensional for Grayscale to RGB conversion.

       \ingroup image_func_gray2rgb
     */
    AFAPI af_err af_gray2rgb(af_array* out, const af_array in, const float rFactor, const float gFactor, const float bFactor);

    /**
       C Interface

       \param[out] out is an array with data that has histogram approximately equal to histogram
       \param[in]  in is the input array, non-normalized input (!! assumes values [0-255] !!)
       \param[in]  hist target histogram to approximate in output (based on # of bins)
       \return     \ref AF_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note \p in must be two dimensional.

       \ingroup image_func_histequal
     */
    AFAPI af_err af_histequal(af_array *out, const af_array in, const af_array hist);

    AFAPI af_err af_gaussian_kernel(af_array *out,
                                    const int rows, const int cols,
                                    const double sigma_r, const double sigma_c);

    /**
       C Interface

       \param[out] out is an array in the RGB colorspace
       \param[in]  in is an array in the HSV colorspace
       \return     \ref AF_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note \p in must be three dimensional

       \ingroup image_func_hsv2rgb
     */
    AFAPI af_err af_hsv2rgb(af_array* out, const af_array in);

    /**
       C Interface

       \param[out] out is an array in the HSV colorspace
       \param[in]  in is an array in the RGB colorspace
       \return     \ref AF_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note \p in must be three dimensional

       \ingroup image_func_rgb2hsv
     */
    AFAPI af_err af_rgb2hsv(af_array* out, const af_array in);

    /**
       C Interface

       \param[out] out is an array in target colorspace \param[in]  image is
       the input array \param[in]  to is the target array colorspace \param[in]
       from is the input array colorspace \return     \ref AF_SUCCESS if the
       color transformation is successful, otherwise an appropriate error code
       is returned.

       \note  \p image must be 3 dimensional for \ref AF_HSV to \ref AF_RGB, \ref
       AF_RGB to \ref AF_HSV, & \ref AF_RGB to \ref AF_GRAY transformations.
       For \ref AF_GRAY to \ref AF_RGB transformation, 2D array is expected.

       \ingroup image_func_colorspace
     */
    AFAPI af_err af_colorspace(af_array *out, const af_array image, af_cspace_t to, af_cspace_t from);

#ifdef __cplusplus
}
#endif
