/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "symbol_manager.hpp"
#include <af/array.h>
#include <af/image.h>
#include <af/defines.h>

af_err af_gradient(af_array *dx, af_array *dy, const af_array in)
{
    CHECK_ARRAYS(in);
    return CALL(dx, dy, in);
}

af_err af_load_image(af_array *out, const char* filename, const bool isColor)
{
    return CALL(out, filename, isColor);
}

af_err af_save_image(const char* filename, const af_array in)
{
    CHECK_ARRAYS(in);
    return CALL(filename, in);
}

af_err af_load_image_memory(af_array *out, const void* ptr)
{
    return CALL(out, ptr);
}

af_err af_save_image_memory(void** ptr, const af_array in, const af_image_format format)
{
    CHECK_ARRAYS(in);
    return CALL(ptr, in, format);
}

af_err af_delete_image_memory(void* ptr)
{
    return CALL(ptr);
}

af_err af_load_image_native(af_array *out, const char* filename)
{
    return CALL(out, filename);
}

af_err af_save_image_native(const char* filename, const af_array in)
{
    CHECK_ARRAYS(in);
    return CALL(filename, in);
}

af_err af_is_image_io_available(bool *out)
{
    return CALL(out);
}

af_err af_resize(af_array *out, const af_array in, const dim_t odim0, const dim_t odim1, const af_interp_type method)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, odim0, odim1, method);
}

af_err af_transform(af_array *out, const af_array in, const af_array transform,
        const dim_t odim0, const dim_t odim1,
        const af_interp_type method, const bool inverse)
{
    CHECK_ARRAYS(in, transform);
    return CALL(out, in, transform, odim0, odim1, method, inverse);
}

af_err af_transform_coordinates(af_array *out, const af_array tf,
        const float d0, const float d1)
{
    CHECK_ARRAYS(tf);
    return CALL(out, tf, d0, d1);
}

af_err af_rotate(af_array *out, const af_array in, const float theta,
        const bool crop, const af_interp_type method)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, theta, crop, method);
}

af_err af_translate(af_array *out, const af_array in, const float trans0, const float trans1,
        const dim_t odim0, const dim_t odim1, const af_interp_type method)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, trans0, trans1, odim0, odim1, method);
}

af_err af_scale(af_array *out, const af_array in, const float scale0, const float scale1,
        const dim_t odim0, const dim_t odim1, const af_interp_type method)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, scale0, scale1, odim0, odim1, method);
}

af_err af_skew(af_array *out, const af_array in, const float skew0, const float skew1,
        const dim_t odim0, const dim_t odim1, const af_interp_type method,
        const bool inverse)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, skew0, skew1, odim0, odim1, method, inverse);
}

af_err af_histogram(af_array *out, const af_array in, const unsigned nbins, const double minval, const double maxval)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, nbins, minval, maxval);
}

af_err af_dilate(af_array *out, const af_array in, const af_array mask)
{
    CHECK_ARRAYS(in, mask);
    return CALL(out, in, mask);
}

af_err af_dilate3(af_array *out, const af_array in, const af_array mask)
{
    CHECK_ARRAYS(in, mask);
    return CALL(out, in, mask);
}

af_err af_erode(af_array *out, const af_array in, const af_array mask)
{
    CHECK_ARRAYS(in, mask);
    return CALL(out, in, mask);
}

af_err af_erode3(af_array *out, const af_array in, const af_array mask)
{
    CHECK_ARRAYS(in, mask);
    return CALL(out, in, mask);
}

af_err af_bilateral(af_array *out, const af_array in, const float spatial_sigma, const float chromatic_sigma, const bool isColor)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, spatial_sigma, chromatic_sigma, isColor);
}

af_err af_mean_shift(af_array *out, const af_array in, const float spatial_sigma, const float chromatic_sigma, const unsigned iter, const bool is_color)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, spatial_sigma, chromatic_sigma, iter, is_color);
}

af_err af_minfilt(af_array *out, const af_array in, const dim_t wind_length, const dim_t wind_width, const af_border_type edge_pad)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, wind_length, wind_width, edge_pad);
}

af_err af_maxfilt(af_array *out, const af_array in, const dim_t wind_length, const dim_t wind_width, const af_border_type edge_pad)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, wind_length, wind_width, edge_pad);
}

af_err af_regions(af_array *out, const af_array in, const af_connectivity connectivity, const af_dtype ty)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, connectivity, ty);
}

af_err af_sobel_operator(af_array *dx, af_array *dy, const af_array img, const unsigned ker_size)
{
    CHECK_ARRAYS(img);
    return CALL(dx, dy, img, ker_size);
}

af_err af_rgb2gray(af_array* out, const af_array in, const float rPercent, const float gPercent, const float bPercent)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, rPercent, gPercent, bPercent);
}

af_err af_gray2rgb(af_array* out, const af_array in, const float rFactor, const float gFactor, const float bFactor)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, rFactor, gFactor, bFactor);
}

af_err af_hist_equal(af_array *out, const af_array in, const af_array hist)
{
    CHECK_ARRAYS(in, hist);
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
    CHECK_ARRAYS(in);
    return CALL(out, in);
}

af_err af_rgb2hsv(af_array* out, const af_array in)
{
    CHECK_ARRAYS(in);
    return CALL(out, in);
}

af_err af_color_space(af_array *out, const af_array image, const af_cspace_t to, const af_cspace_t from)
{
    CHECK_ARRAYS(image);
    return CALL(out, image, to, from);
}

af_err af_unwrap(af_array *out, const af_array in, const dim_t wx, const dim_t wy,
        const dim_t sx, const dim_t sy, const dim_t px, const dim_t py,
        const bool is_column)
{
    CHECK_ARRAYS(in);
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
    CHECK_ARRAYS(in);
    return CALL(out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);
}

af_err af_sat(af_array *out, const af_array in)
{
    CHECK_ARRAYS(in);
    return CALL(out, in);
}

af_err af_ycbcr2rgb(af_array* out, const af_array in, const af_ycc_std standard)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, standard);
}

af_err af_rgb2ycbcr(af_array* out, const af_array in, const af_ycc_std standard)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, standard);
}

af_err af_canny(af_array* out, const af_array in, const af_canny_threshold ct,
                const float t1, const float t2, const unsigned sw, const bool isf)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, ct, t1, t2, sw, isf);
}

af_err af_anisotropic_diffusion(af_array* out, const af_array in, const float dt,
                                const float K, const unsigned iterations,
                                const af_flux_function fftype,
                                const af_diffusion_eq eq)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, dt, K, iterations, fftype, eq);
}

af_err af_iterative_deconv(af_array* out, const af_array in, const af_array ker,
                           const unsigned iterations, const float relax_factor,
                           const af_iterative_deconv_algo algo)
{
    CHECK_ARRAYS(in, ker);
    return CALL(out, in, ker, iterations, relax_factor, algo);
}

af_err af_inverse_deconv(af_array* out, const af_array in, const af_array psf,
                         const float gamma,const af_inverse_deconv_algo algo)
{
    CHECK_ARRAYS(in, psf);
    return CALL(out, in, psf, gamma, algo);
}
