/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/features.h>

#ifdef __cplusplus
namespace af
{
class array;

/**
   C++ Interface for calculating the gradients

   \param[out] dx the gradient along first dimension
   \param[out] dy the gradient along second dimension
   \param[in] in is the input array

   \ingroup calc_func_grad
*/
AFAPI void grad(array& dx, array& dy, const array& in);

/**
    C++ Interface for loading an image

    \param[in] filename is name of file to be loaded
    \param[in] is_color boolean denoting if the image should be loaded as 1 channel or 3 channel
    \return image loaded as \ref af::array()

    \ingroup imageio_func_load
*/
AFAPI array loadImage(const char* filename, const bool is_color=false);

/**
    C++ Interface for saving an image

    \param[in] filename is name of file to be loaded
    \param[in] in is the arrayfire array to be saved as an image

    \ingroup imageio_func_save
*/
AFAPI void saveImage(const char* filename, const array& in);

/**
    C++ Interface for resizing an image to specified dimensions

    \param[in] in is input image
    \param[in] odim0 is the size for the first output dimension
    \param[in] odim1 is the size for the second output dimension
    \param[in] method is the interpolation type (Nearest by default)
    \return the resized image of specified by \p odim0 and \p odim1

    \ingroup transform_func_resize
*/
AFAPI array resize(const array& in, const dim_t odim0, const dim_t odim1, const interpType method=AF_INTERP_NEAREST);

/**
    C++ Interface for resizing an image to specified scales

    \param[in] scale0 is scale used for first input dimension
    \param[in] scale1 is scale used for second input dimension
    \param[in] in is input image
    \param[in] method is the interpolation type (Nearest by default)
    \return the image scaled by the specified by \p scale0 and \p scale1

    \ingroup transform_func_resize
*/
AFAPI array resize(const float scale0, const float scale1, const array& in, const interpType method=AF_INTERP_NEAREST);

/**
    C++ Interface for resizing an image to specified scale

    \param[in] scale is scale used for both input dimensions
    \param[in] in is input image
    \param[in] method is the interpolation type (Nearest by default)
    \return the image scaled by the specified by \p scale

    \ingroup transform_func_resize
*/
AFAPI array resize(const float scale, const array& in, const interpType method=AF_INTERP_NEAREST);

/**
    C++ Interface for rotating an image

    \param[in] in is input image
    \param[in] theta is the degree (in radians) by which the input is rotated
    \param[in] crop if true the output is cropped original dimensions. If false the output dimensions scale based on \p theta
    \param[in] method is the interpolation type (Nearest by default)
    \return the image rotated by \p theta

    \ingroup transform_func_rotate
*/
AFAPI array rotate(const array& in, const float theta, const bool crop=true, const interpType method=AF_INTERP_NEAREST);

/**
    C++ Interface for transforming an image

    \param[in] in is input image
    \param[in] transform is transformation matrix
    \param[in] odim0 is the first output dimension
    \param[in] odim1 is the second output dimension
    \param[in] method is the interpolation type (Nearest by default)
    \param[in] inverse if true applies inverse transform, if false applies forward transoform
    \return the transformed image

    \ingroup transform_func_transform
*/
AFAPI array transform(const array& in, const array& transform, const dim_t odim0 = 0, const dim_t odim1 = 0, const interpType method=AF_INTERP_NEAREST, const bool inverse=true);

/**
    C++ Interface for translating an image

    \param[in] in is input image
    \param[in] trans0 is amount by which the first dimension is translated
    \param[in] trans1 is amount by which the second dimension is translated
    \param[in] odim0 is the first output dimension
    \param[in] odim1 is the second output dimension
    \param[in] method is the interpolation type (Nearest by default)
    \return the translated image

    \ingroup transform_func_translate
*/
AFAPI array translate(const array& in, const float trans0, const float trans1, const dim_t odim0 = 0, const dim_t odim1 = 0, const interpType method=AF_INTERP_NEAREST);

/**
    C++ Interface for scaling an image

    \param[in] in is input image
    \param[in] scale0 is amount by which the first dimension is scaled
    \param[in] scale1 is amount by which the second dimension is scaled
    \param[in] odim0 is the first output dimension
    \param[in] odim1 is the second output dimension
    \param[in] method is the interpolation type (Nearest by default)
    \return the scaled image

    \ingroup transform_func_scale
*/
AFAPI array scale(const array& in, const float scale0, const float scale1, const dim_t odim0 = 0, const dim_t odim1 = 0, const interpType method=AF_INTERP_NEAREST);

/**
    C++ Interface for skewing an image

    \param[in] in is input image
    \param[in] skew0 is amount by which the first dimension is skewed
    \param[in] skew1 is amount by which the second dimension is skewed
    \param[in] odim0 is the first output dimension
    \param[in] odim1 is the second output dimension
    \param[in] inverse if true applies inverse transform, if false applies forward transoform
    \param[in] method is the interpolation type (Nearest by default)
    \return the skewed image

    \ingroup transform_func_skew
*/
AFAPI array skew(const array& in, const float skew0, const float skew1, const dim_t odim0 = 0, const dim_t odim1 = 0, const bool inverse=true, const interpType method=AF_INTERP_NEAREST);

/**
    C++ Interface for bilateral filter

    \param[in]  in array is the input image
    \param[in]  spatial_sigma is the spatial variance parameter that decides the filter window
    \param[in]  chromatic_sigma is the chromatic variance parameter
    \param[in]  is_color indicates if the input \p in is color image or grayscale
    \return     the processed image

    \ingroup image_func_bilateral
*/
AFAPI array bilateral(const array &in, const float spatial_sigma, const float chromatic_sigma, const bool is_color=false);

/**
   C++ Interface for histogram

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
   C++ Interface for histogram

   \snippet test/histogram.cpp ex_image_hist_nominmax

   \param[in]  in is the input array
   \param[in]  nbins  Number of bins to populate between min and max
   \return     histogram array

   \ingroup image_func_histogram
 */
AFAPI array histogram(const array &in, const unsigned nbins);

/**
    C++ Interface for mean shift

    \param[in]  in array is the input image
    \param[in]  spatial_sigma is the spatial variance parameter that decides the filter window
    \param[in]  chromatic_sigma is the chromatic variance parameter
    \param[in]  iter is the number of iterations filter operation is performed
    \param[in]  is_color indicates if the input \p in is color image or grayscale
    \return     the processed image

    \ingroup image_func_mean_shift
*/
AFAPI array meanShift(const array& in, const float spatial_sigma, const float chromatic_sigma, const unsigned iter, const bool is_color=false);

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

/**
    C++ Interface for minimum filter

    \param[in]  in array is the input image
    \param[in]  wind_length is the kernel height
    \param[in]  wind_width is the kernel width
    \param[in]  edge_pad value will decide what happens to border when running
                filter in their neighborhood. It takes one of the values [\ref AF_PAD_ZERO | \ref AF_PAD_SYM]
    \return     the processed image

    \ingroup image_func_minfilt
*/
AFAPI array minfilt(const array& in, const dim_t wind_length = 3, const dim_t wind_width = 3, const borderType edge_pad = AF_PAD_ZERO);

/**
    C++ Interface for maximum filter

    \param[in]  in array is the input image
    \param[in]  wind_length is the kernel height
    \param[in]  wind_width is the kernel width
    \param[in]  edge_pad value will decide what happens to border when running
                filter in their neighborhood. It takes one of the values [\ref AF_PAD_ZERO | \ref AF_PAD_SYM]
    \return     the processed image

    \ingroup image_func_maxfilt
*/
AFAPI array maxfilt(const array& in, const dim_t wind_length = 3, const dim_t wind_width = 3, const borderType edge_pad = AF_PAD_ZERO);

/**
    C++ Interface for image dilation (max filter)

    \param[in]  in array is the input image
    \param[in]  mask is the neighborhood window
    \return     the dilated image

    \note if \p mask is all ones, this function behaves like max filter

    \ingroup image_func_dilate
*/
AFAPI array dilate(const array& in, const array& mask);

/**
    C++ Interface for 3D image dilation

    \param[in]  in array is the input volume
    \param[in]  mask is the neighborhood delta volume
    \return     the dilated volume

    \ingroup image_func_dilate3d
*/
AFAPI array dilate3(const array& in, const array& mask);

/**
    C++ Interface for image erosion (min filter)

    \param[in]  in array is the input image
    \param[in]  mask is the neighborhood window
    \return     the eroded image

    \note This function can be used as min filter by using a mask of all ones

    \ingroup image_func_erode
*/
AFAPI array erode(const array& in, const array& mask);

/**
    C++ Interface for 3d for image erosion

    \param[in]  in array is the input volume
    \param[in]  mask is the neighborhood delta volume
    \return     the eroded volume

    \ingroup image_func_erode3d
*/
AFAPI array erode3(const array& in, const array& mask);

/**
    C++ Interface for getting regions in an image

    Below given are sample input and output for each type of connectivity value for \p type

    <table border="0">
    <tr>
    <td> Example for \p type == \ref AF_CONNECTIVITY_8 </td>
    <td> Example for \p type == \ref AF_CONNECTIVITY_4 </td>
    </tr>
    <tr>
    <td>
        \snippet test/regions.cpp ex_image_regions
    </td>
    <td>
        \snippet test/regions.cpp ex_image_regions_4conn
    </td>
    </tr>
    </table>

    \param[in]  in array should be binary image of type \ref b8
    \param[in]  connectivity can take one of the following [\ref AF_CONNECTIVITY_4 | \ref AF_CONNECTIVITY_8]
    \param[in]  type is type of output array
    \return     returns array with labels indicating different regions. Throws exceptions if any issue occur.

    \ingroup image_func_regions
*/
AFAPI array regions(const array& in, const af::connectivity connectivity=AF_CONNECTIVITY_4, const dtype type=f32);

/**
   C++ Interface for extracting sobel gradients

   \param[out] dx is derivative along horizontal direction
   \param[out] dy is derivative along vertical direction
   \param[in]  img is an array with image data
   \param[in]  ker_size sobel kernel size or window size

   \note If \p img is 3d array, a batch operation will be performed.

   \ingroup image_func_sobel
 */
AFAPI void sobel(array &dx, array &dy, const array &img, const unsigned ker_size=3);

/**
   C++ Interface for sobel filtering

   \param[in]  img is an array with image data
   \param[in]  ker_size sobel kernel size or window size
   \param[in]  isFast = true uses \f$G=G_x+G_y\f$, otherwise \f$G=\sqrt (G_x^2+G_y^2)\f$
   \return     an array with sobel gradient values

   \note If \p img is 3d array, a batch operation will be performed.

   \ingroup image_func_sobel
 */
AFAPI array sobel(const array &img, const unsigned ker_size=3, const bool isFast=false);

/**
   C++ Interface for RGB to gray conversion

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
   C++ Interface for gray to RGB conversion

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
   C++ Interface for histogram equalization

   \snippet test/histogram.cpp ex_image_histequal

   \param[in]  in is the input array, non-normalized input (!! assumes values [0-255] !!)
   \param[in]  hist target histogram to approximate in output (based on number of bins)
   \return     data with histogram approximately equal to histogram

   \note \p in must be two dimensional.

   \ingroup image_func_histequal
 */
AFAPI array histEqual(const array& in, const array& hist);

/**
   C++ Interface for generating gausian kernels

   \param[in]  rows number of kernel rows
   \param[in]  cols number of kernel columns
   \param[in]  sig_r (default 0) (calculated internally as 0.25 * rows + 0.75)
   \param[in]  sig_c (default 0) (calculated internally as 0.25 * cols + 0.75)
   \return     an array with values generated using gaussian function

   \ingroup image_func_gauss
 */
AFAPI array gaussianKernel(const int rows, const int cols, const double sig_r = 0, const double sig_c = 0);

/**
   C++ Interface for converting HSV to RGB

   \param[in]  in is an array in the HSV colorspace
   \return     array in RGB colorspace

   \note \p in must be three dimensional

   \ingroup image_func_hsv2rgb
 */
AFAPI array hsv2rgb(const array& in);

/**
   C++ Interface for converting RGB to HSV

   \param[in]  in is an array in the RGB colorspace
   \return     array in HSV colorspace

   \note \p in must be three dimensional

   \ingroup image_func_rgb2hsv
 */
AFAPI array rgb2hsv(const array& in);

/**
   C++ Interface wrapper for colorspace conversion

   \param[in]  image is the input array
   \param[in]  to is the target array colorspace
   \param[in]  from is the input array colorspace
   \return     array in target colorspace

   \note  \p image must be 3 dimensional for \ref AF_HSV to \ref AF_RGB, \ref AF_RGB to
   \ref AF_HSV, & \ref AF_RGB to \ref AF_GRAY transformations. For \ref AF_GRAY to \ref AF_RGB
   transformation, 2D array is expected.

   \ingroup image_func_colorspace
 */
AFAPI array colorSpace(const array& image, const CSpace to, const CSpace from);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       C Interface for calculating the gradients

       \param[out] dx the gradient along first dimension
       \param[out] dy the gradient along second dimension
       \param[in]  in is the input array
       \return     \ref AF_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \ingroup calc_func_grad
    */
    AFAPI af_err af_gradient(af_array *dx, af_array *dy, const af_array in);

    /**
       C Interface for loading an image

       \param[out] out will contain the image
       \param[in] filename is name of file to be loaded
       \param[in] isColor boolean denoting if the image should be loaded as 1 channel or 3 channel
       \return     \ref AF_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \ingroup imageio_func_load
    */
    AFAPI af_err af_load_image(af_array *out, const char* filename, const bool isColor);

   /**
      C Interface for saving an image

      \param[in] filename is name of file to be loaded
      \param[in] in is the arrayfire array to be saved as an image
      \return     \ref AF_SUCCESS if the color transformation is successful,
      otherwise an appropriate error code is returned.

      \ingroup imageio_func_save
   */
    AFAPI af_err af_save_image(const char* filename, const af_array in);

    /**
       C Interface for resizing an image to specified dimensions

       \param[out] out will contain the resized image of specified by \p odim0 and \p odim1
       \param[in] in is input image
       \param[in] odim0 is the size for the first output dimension
       \param[in] odim1 is the size for the second output dimension
       \param[in] method is the interpolation type (Nearest by default)

       \return     \ref AF_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \ingroup transform_func_resize
    */
    AFAPI af_err af_resize(af_array *out, const af_array in, const dim_t odim0, const dim_t odim1, const af_interp_type method);

    /**
       C Interface for transforming an image

       \param[out] out will contain the transformed image
       \param[in] in is input image
       \param[in] transform is transformation matrix
       \param[in] odim0 is the first output dimension
       \param[in] odim1 is the second output dimension
       \param[in] method is the interpolation type (Nearest by default)
       \param[in] inverse if true applies inverse transform, if false applies forward transoform
       \return     \ref AF_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \ingroup transform_func_transform
    */
    AFAPI af_err af_transform(af_array *out, const af_array in, const af_array transform,
                              const dim_t odim0, const dim_t odim1,
                              const af_interp_type method, const bool inverse);

    /**
       C Interface for rotating an image

       \param[out] out will contain the image \p in rotated by \p theta
       \param[in] in is input image
       \param[in] theta is the degree (in radians) by which the input is rotated
       \param[in] crop if true the output is cropped original dimensions. If false the output dimensions scale based on \p theta
       \param[in] method is the interpolation type (Nearest by default)
       \return     \ref AF_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \ingroup transform_func_rotate
    */
    AFAPI af_err af_rotate(af_array *out, const af_array in, const float theta,
                           const bool crop, const af_interp_type method);
   /**
      C Interface for translate an image

      \param[out] out will contain the translated image
      \param[in] in is input image
      \param[in] trans0 is amount by which the first dimension is translated
      \param[in] trans1 is amount by which the second dimension is translated
      \param[in] odim0 is the first output dimension
      \param[in] odim1 is the second output dimension
      \param[in] method is the interpolation type (Nearest by default)
      \return     \ref AF_SUCCESS if the color transformation is successful,
      otherwise an appropriate error code is returned.

      \ingroup transform_func_translate
   */
    AFAPI af_err af_translate(af_array *out, const af_array in, const float trans0, const float trans1,
                              const dim_t odim0, const dim_t odim1, const af_interp_type method);
    /**
       C Interface for scaling an image

       \param[out] out will contain the scaled image
       \param[in] in is input image
       \param[in] scale0 is amount by which the first dimension is scaled
       \param[in] scale1 is amount by which the second dimension is scaled
       \param[in] odim0 is the first output dimension
       \param[in] odim1 is the second output dimension
       \param[in] method is the interpolation type (Nearest by default)
       \return     \ref AF_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \ingroup transform_func_scale
    */
    AFAPI af_err af_scale(af_array *out, const af_array in, const float scale0, const float scale1,
                          const dim_t odim0, const dim_t odim1, const af_interp_type method);
    /**
       C Interface for skewing an image

       \param[out] out will contain the skewed image
       \param[in] in is input image
       \param[in] skew0 is amount by which the first dimension is skewed
       \param[in] skew1 is amount by which the second dimension is skewed
       \param[in] odim0 is the first output dimension
       \param[in] odim1 is the second output dimension
       \param[in] inverse if true applies inverse transform, if false applies forward transoform
       \param[in] method is the interpolation type (Nearest by default)
       \return     \ref AF_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \ingroup transform_func_skew
    */
    AFAPI af_err af_skew(af_array *out, const af_array in, const float skew0, const float skew1,
                         const dim_t odim0, const dim_t odim1, const af_interp_type method,
                         const bool inverse);

    /**
       C Interface for histogram

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

    /**
        C Interface for image dilation (max filter)

        \param[out] out array is the dilated image
        \param[in]  in array is the input image
        \param[in]  mask is the neighborhood window
        \return     \ref AF_SUCCESS if the dilated successfully,
        otherwise an appropriate error code is returned.

        \note if \p mask is all ones, this function behaves like max filter

        \ingroup image_func_dilate
    */
    AFAPI af_err af_dilate(af_array *out, const af_array in, const af_array mask);

    /**
        C Interface for 3d image dilation

        \param[out] out array is the dilated volume
        \param[in]  in array is the input volume
        \param[in]  mask is the neighborhood delta volume
        \return     \ref AF_SUCCESS if the dilated successfully,
        otherwise an appropriate error code is returned.

        \ingroup image_func_dilate3d
    */
    AFAPI af_err af_dilate3(af_array *out, const af_array in, const af_array mask);

    /**
        C Interface for image erosion (min filter)

        \param[out] out array is the eroded image
        \param[in]  in array is the input image
        \param[in]  mask is the neighborhood window
        \return     \ref AF_SUCCESS if the eroded successfully,
        otherwise an appropriate error code is returned.

        \note if \p mask is all ones, this function behaves like min filter

        \ingroup image_func_erode
    */
    AFAPI af_err af_erode(af_array *out, const af_array in, const af_array mask);

    /**
        C Interface for 3D image erosion

        \param[out] out array is the eroded volume
        \param[in]  in array is the input volume
        \param[in]  mask is the neighborhood delta volume
        \return     \ref AF_SUCCESS if the eroded successfully,
        otherwise an appropriate error code is returned.

        \ingroup image_func_erode3d
    */
    AFAPI af_err af_erode3(af_array *out, const af_array in, const af_array mask);

    /**
        C Interface for bilateral filter

        \param[out] out array is the processed image
        \param[in]  in array is the input image
        \param[in]  spatial_sigma is the spatial variance parameter that decides the filter window
        \param[in]  chromatic_sigma is the chromatic variance parameter
        \param[in]  isColor indicates if the input \p in is color image or grayscale
        \return     \ref AF_SUCCESS if the filter is applied successfully,
        otherwise an appropriate error code is returned.

        \ingroup image_func_bilateral
    */
    AFAPI af_err af_bilateral(af_array *out, const af_array in, const float spatial_sigma, const float chromatic_sigma, const bool isColor);

    /**
        C Interface for mean shift

        \param[out] out array is the processed image
        \param[in]  in array is the input image
        \param[in]  spatial_sigma is the spatial variance parameter that decides the filter window
        \param[in]  chromatic_sigma is the chromatic variance parameter
        \param[in]  iter is the number of iterations filter operation is performed
        \param[in]  is_color indicates if the input \p in is color image or grayscale
        \return     \ref AF_SUCCESS if the filter is applied successfully,
        otherwise an appropriate error code is returned.

        \ingroup image_func_mean_shift
    */
    AFAPI af_err af_mean_shift(af_array *out, const af_array in, const float spatial_sigma, const float chromatic_sigma, const unsigned iter, const bool is_color);

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

    /**
        C Interface for minimum filter

        \param[out] out array is the processed image
        \param[in]  in array is the input image
        \param[in]  wind_length is the kernel height
        \param[in]  wind_width is the kernel width
        \param[in]  edge_pad value will decide what happens to border when running
                    filter in their neighborhood. It takes one of the values [\ref AF_PAD_ZERO | \ref AF_PAD_SYM]
        \return     \ref AF_SUCCESS if the minimum filter is applied successfully,
        otherwise an appropriate error code is returned.

        \ingroup image_func_minfilt
    */
    AFAPI af_err af_minfilt(af_array *out, const af_array in, const dim_t wind_length, const dim_t wind_width, const af_border_type edge_pad);

    /**
       C Interface for maximum filter

       \param[out] out array is the processed image
       \param[in]  in array is the input image
       \param[in]  wind_length is the kernel height
       \param[in]  wind_width is the kernel width
       \param[in]  edge_pad value will decide what happens to border when running
       filter in their neighborhood. It takes one of the values [\ref AF_PAD_ZERO | \ref AF_PAD_SYM]
       \return     \ref AF_SUCCESS if the maximum filter is applied successfully,
       otherwise an appropriate error code is returned.

       \ingroup image_func_maxfilt
    */
    AFAPI af_err af_maxfilt(af_array *out, const af_array in, const dim_t wind_length, const dim_t wind_width, const af_border_type edge_pad);

    /**
        C Interface for regions in an image

        \param[out] out array will have labels indicating different regions
        \param[in]  in array should be binary image of type \ref b8
        \param[in]  connectivity can take one of the following [\ref AF_CONNECTIVITY_4 | \ref AF_CONNECTIVITY_8]
        \param[in]  ty is type of output array
        \return     \ref AF_SUCCESS if the regions are identified successfully,
        otherwise an appropriate error code is returned.

        \ingroup image_func_regions
    */
    AFAPI af_err af_regions(af_array *out, const af_array in, const af_connectivity connectivity, const af_dtype ty);

    /**
       C Interface for getting sobel gradients

       \param[out] dx is derivative along horizontal direction
       \param[out] dy is derivative along vertical direction
       \param[in]  img is an array with image data
       \param[in]  ker_size sobel kernel size or window size
       \return     \ref AF_SUCCESS if sobel derivatives are computed successfully,
       otherwise an appropriate error code is returned.

       \note If \p img is 3d array, a batch operation will be performed.

       \ingroup image_func_sobel
    */
    AFAPI af_err af_sobel_operator(af_array *dx, af_array *dy, const af_array img, const unsigned ker_size);

    /**
       C Interface for converting RGB to gray

       \param[out] out is an array in target color space
       \param[in]  in is an array in the RGB color space
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
       C Interface for converting gray to RGB

       \param[out] out is an array in target color space
       \param[in]  in is an array in the Grayscale color space
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
       C Interface for histogram equalization

       \param[out] out is an array with data that has histogram approximately equal to histogram
       \param[in]  in is the input array, non-normalized input (!! assumes values [0-255] !!)
       \param[in]  hist target histogram to approximate in output (based on number of bins)
       \return     \ref AF_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note \p in must be two dimensional.

       \ingroup image_func_histequal
    */
    AFAPI af_err af_hist_equal(af_array *out, const af_array in, const af_array hist);

    /**
       C Interface generating gaussian kernels

       \param[out] out is an array with values generated using gaussian function
       \param[in]  rows number of kernel rows
       \param[in]  cols number of kernel columns
       \param[in]  sigma_r (default 0) (calculated internally as 0.25 * rows + 0.75)
       \param[in]  sigma_c (default 0) (calculated internally as 0.25 * cols + 0.75)
       \return     \ref AF_SUCCESS if gaussian distribution values are generated successfully,
       otherwise an appropriate error code is returned.

       \ingroup image_func_gauss
    */
    AFAPI af_err af_gaussian_kernel(af_array *out,
                                    const int rows, const int cols,
                                    const double sigma_r, const double sigma_c);

    /**
       C Interface for converting HSV to RGB

       \param[out] out is an array in the RGB color space
       \param[in]  in is an array in the HSV color space
       \return     \ref AF_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note \p in must be three dimensional

       \ingroup image_func_hsv2rgb
    */
    AFAPI af_err af_hsv2rgb(af_array* out, const af_array in);

    /**
       C Interface for converting RGB to HSV

       \param[out] out is an array in the HSV color space
       \param[in]  in is an array in the RGB color space
       \return     \ref AF_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note \p in must be three dimensional

       \ingroup image_func_rgb2hsv
    */
    AFAPI af_err af_rgb2hsv(af_array* out, const af_array in);

    /**
       C Interface wrapper for color space conversion

       \param[out] out is an array in target color space
       \param[in]  image is the input array
       \param[in]  to is the target array color space \param[in]
       from is the input array color space
       \return     \ref AF_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code
       is returned.

       \note  \p image must be 3 dimensional for \ref AF_HSV to \ref AF_RGB, \ref
       AF_RGB to \ref AF_HSV, & \ref AF_RGB to \ref AF_GRAY transformations.
       For \ref AF_GRAY to \ref AF_RGB transformation, 2D array is expected.

       \ingroup image_func_colorspace
    */
    AFAPI af_err af_color_space(af_array *out, const af_array image, const af_cspace_t to, const af_cspace_t from);

#ifdef __cplusplus
}
#endif
