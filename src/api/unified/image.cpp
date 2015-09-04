/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/image.h>
#include "symbol_manager.hpp"

af_err af_gradient(af_array *dx, af_array *dy, const af_array in)
{
    return CALL(dx, dy, in);
}

af_err af_load_image(af_array *out, const char* filename, const bool isColor)
{
    return CALL(out, filename, isColor);
}

af_err af_save_image(const char* filename, const af_array in)
{
    return CALL(in);
}

af_err af_load_image_memory(af_array *out, const void* ptr)
{
    return CALL(out, ptr);
}

af_err af_save_image_memory(void** ptr, const af_array in, const af_image_format format)
{
    return CALL(ptr, in, format);
}

af_err af_delete_image_memory(void* ptr)
{
    return CALL(ptr);
}

af_err af_resize(af_array *out, const af_array in, const dim_t odim0, const dim_t odim1, const af_interp_type method)
{
    return CALL(out, in, odim0, odim1, method);
}

af_err af_transform(af_array *out, const af_array in, const af_array transform,
        const dim_t odim0, const dim_t odim1,
        const af_interp_type method, const bool inverse)
{
    return CALL(out, in, transform, odim0, odim1, method, inverse);
}

af_err af_rotate(af_array *out, const af_array in, const float theta,
        const bool crop, const af_interp_type method)
{
    return CALL(out, in, theta, crop, method);
}

af_err af_translate(af_array *out, const af_array in, const float trans0, const float trans1,
        const dim_t odim0, const dim_t odim1, const af_interp_type method)
{
    return CALL(out, in, trans0, trans1, odim0, odim1, method);
}

af_err af_scale(af_array *out, const af_array in, const float scale0, const float scale1,
        const dim_t odim0, const dim_t odim1, const af_interp_type method)
{
    return CALL(out, in, scale0, scale1, odim0, odim1, method);
}

af_err af_skew(af_array *out, const af_array in, const float skew0, const float skew1,
        const dim_t odim0, const dim_t odim1, const af_interp_type method,
        const bool inverse)
{
    return CALL(out, in, skew0, skew1, odim0, odim1, method, inverse);
}

af_err af_histogram(af_array *out, const af_array in, const unsigned nbins, const double minval, const double maxval)
{
    return CALL(out, in, nbins, minval, maxval);
}

af_err af_dilate(af_array *out, const af_array in, const af_array mask)
{
    return CALL(out, in, mask);
}

af_err af_dilate3(af_array *out, const af_array in, const af_array mask)
{
    return CALL(out, in, mask);
}

af_err af_erode(af_array *out, const af_array in, const af_array mask)
{
    return CALL(out, in, mask);
}

af_err af_erode3(af_array *out, const af_array in, const af_array mask)
{
    return CALL(out, in, mask);
}

af_err af_bilateral(af_array *out, const af_array in, const float spatial_sigma, const float chromatic_sigma, const bool isColor)
{
    return CALL(out, in, spatial_sigma, chromatic_sigma, isColor);
}

af_err af_mean_shift(af_array *out, const af_array in, const float spatial_sigma, const float chromatic_sigma, const unsigned iter, const bool is_color)
{
    return CALL(out, in, spatial_sigma, chromatic_sigma, iter, is_color);
}

af_err af_medfilt(af_array *out, const af_array in, const dim_t wind_length, const dim_t wind_width, const af_border_type edge_pad)
{
    return CALL(out, in, wind_length, wind_width, edge_pad);
}

af_err af_minfilt(af_array *out, const af_array in, const dim_t wind_length, const dim_t wind_width, const af_border_type edge_pad)
{
    return CALL(out, in, wind_length, wind_width, edge_pad);
}

af_err af_maxfilt(af_array *out, const af_array in, const dim_t wind_length, const dim_t wind_width, const af_border_type edge_pad)
{
    return CALL(out, in, wind_length, wind_width, edge_pad);
}

af_err af_regions(af_array *out, const af_array in, const af_connectivity connectivity, const af_dtype ty)
{
    return CALL(out, in, connectivity, ty);
}

af_err af_sobel_operator(af_array *dx, af_array *dy, const af_array img, const unsigned ker_size)
{
    return CALL(dx, dy, img, ker_size);
}

af_err af_rgb2gray(af_array* out, const af_array in, const float rPercent, const float gPercent, const float bPercent)
{
    return CALL(out, in, rPercent, gPercent, bPercent);
}

af_err af_gray2rgb(af_array* out, const af_array in, const float rFactor, const float gFactor, const float bFactor)
{
    return CALL(out, in, rFactor, gFactor, bFactor);
}

af_err af_hist_equal(af_array *out, const af_array in, const af_array hist)
{
    return CALL(out, in, hist);
}

af_err af_gaussian_kernel(af_array *out,
        const int rows, const int cols,
        const double sigma_r, const double sigma_c)
{
    return CALL(out, rows, cols, sigma_r, sigma_c);
}

af_err af_hsv2rgb(af_array* out, const af_array in)
{
    return CALL(out, in);
}

af_err af_rgb2hsv(af_array* out, const af_array in)
{
    return CALL(out, in);
}

af_err af_color_space(af_array *out, const af_array image, const af_cspace_t to, const af_cspace_t from)
{
    return CALL(out, image, to, from);
}

af_err af_unwrap(af_array *out, const af_array in, const dim_t wx, const dim_t wy,
        const dim_t sx, const dim_t sy, const dim_t px, const dim_t py,
        const bool is_column)
{
    return CALL(out, in, wx, wy, sx, sy, px, py, is_column);
}

af_err af_wrap(af_array *out,
        const af_array in,
        const dim_t ox, const dim_t oy,
        const dim_t wx, const dim_t wy,
        const dim_t sx, const dim_t sy,
        const dim_t px, const dim_t py,
        const bool is_column)
{
    return CALL(out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);
}

af_err af_sat(af_array *out, const af_array in)
{
    return CALL(out, in);
}

af_err af_ycbcr2rgb(af_array* out, const af_array in, const af_ycc_std standard)
{
    return CALL(out, in, standard);
}

af_err af_rgb2ycbcr(af_array* out, const af_array in, const af_ycc_std standard)
{
    return CALL(out, in, standard);
}
