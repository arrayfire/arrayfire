/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/defines.h>
#include <af/image.h>
#include "symbol_manager.hpp"

af_err af_gradient(af_array *dx, af_array *dy, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_gradient, dx, dy, in);
}

af_err af_load_image(af_array *out, const char *filename, const bool isColor) {
    CALL(af_load_image, out, filename, isColor);
}

af_err af_save_image(const char *filename, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_save_image, filename, in);
}

af_err af_load_image_memory(af_array *out, const void *ptr) {
    CALL(af_load_image_memory, out, ptr);
}

af_err af_save_image_memory(void **ptr, const af_array in,
                            const af_image_format format) {
    CHECK_ARRAYS(in);
    CALL(af_save_image_memory, ptr, in, format);
}

af_err af_delete_image_memory(void *ptr) { CALL(af_delete_image_memory, ptr); }

af_err af_load_image_native(af_array *out, const char *filename) {
    CALL(af_load_image_native, out, filename);
}

af_err af_save_image_native(const char *filename, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_save_image_native, filename, in);
}

af_err af_is_image_io_available(bool *out) {
    CALL(af_is_image_io_available, out);
}

af_err af_resize(af_array *out, const af_array in, const dim_t odim0,
                 const dim_t odim1, const af_interp_type method) {
    CHECK_ARRAYS(in);
    CALL(af_resize, out, in, odim0, odim1, method);
}

af_err af_transform(af_array *out, const af_array in, const af_array transform,
                    const dim_t odim0, const dim_t odim1,
                    const af_interp_type method, const bool inverse) {
    CHECK_ARRAYS(in, transform);
    CALL(af_transform, out, in, transform, odim0, odim1, method, inverse);
}

af_err af_transform_v2(af_array *out, const af_array in,
                       const af_array transform, const dim_t odim0,
                       const dim_t odim1, const af_interp_type method,
                       const bool inverse) {
    CHECK_ARRAYS(out, in, transform);
    CALL(af_transform_v2, out, in, transform, odim0, odim1, method, inverse);
}

af_err af_transform_coordinates(af_array *out, const af_array tf,
                                const float d0, const float d1) {
    CHECK_ARRAYS(tf);
    CALL(af_transform_coordinates, out, tf, d0, d1);
}

af_err af_rotate(af_array *out, const af_array in, const float theta,
                 const bool crop, const af_interp_type method) {
    CHECK_ARRAYS(in);
    CALL(af_rotate, out, in, theta, crop, method);
}

af_err af_translate(af_array *out, const af_array in, const float trans0,
                    const float trans1, const dim_t odim0, const dim_t odim1,
                    const af_interp_type method) {
    CHECK_ARRAYS(in);
    CALL(af_translate, out, in, trans0, trans1, odim0, odim1, method);
}

af_err af_scale(af_array *out, const af_array in, const float scale0,
                const float scale1, const dim_t odim0, const dim_t odim1,
                const af_interp_type method) {
    CHECK_ARRAYS(in);
    CALL(af_scale, out, in, scale0, scale1, odim0, odim1, method);
}

af_err af_skew(af_array *out, const af_array in, const float skew0,
               const float skew1, const dim_t odim0, const dim_t odim1,
               const af_interp_type method, const bool inverse) {
    CHECK_ARRAYS(in);
    CALL(af_skew, out, in, skew0, skew1, odim0, odim1, method, inverse);
}

af_err af_histogram(af_array *out, const af_array in, const unsigned nbins,
                    const double minval, const double maxval) {
    CHECK_ARRAYS(in);
    CALL(af_histogram, out, in, nbins, minval, maxval);
}

af_err af_dilate(af_array *out, const af_array in, const af_array mask) {
    CHECK_ARRAYS(in, mask);
    CALL(af_dilate, out, in, mask);
}

af_err af_dilate3(af_array *out, const af_array in, const af_array mask) {
    CHECK_ARRAYS(in, mask);
    CALL(af_dilate3, out, in, mask);
}

af_err af_erode(af_array *out, const af_array in, const af_array mask) {
    CHECK_ARRAYS(in, mask);
    CALL(af_erode, out, in, mask);
}

af_err af_erode3(af_array *out, const af_array in, const af_array mask) {
    CHECK_ARRAYS(in, mask);
    CALL(af_erode3, out, in, mask);
}

af_err af_bilateral(af_array *out, const af_array in, const float spatial_sigma,
                    const float chromatic_sigma, const bool isColor) {
    CHECK_ARRAYS(in);
    CALL(af_bilateral, out, in, spatial_sigma, chromatic_sigma, isColor);
}

af_err af_mean_shift(af_array *out, const af_array in,
                     const float spatial_sigma, const float chromatic_sigma,
                     const unsigned iter, const bool is_color) {
    CHECK_ARRAYS(in);
    CALL(af_mean_shift, out, in, spatial_sigma, chromatic_sigma, iter,
         is_color);
}

af_err af_minfilt(af_array *out, const af_array in, const dim_t wind_length,
                  const dim_t wind_width, const af_border_type edge_pad) {
    CHECK_ARRAYS(in);
    CALL(af_minfilt, out, in, wind_length, wind_width, edge_pad);
}

af_err af_maxfilt(af_array *out, const af_array in, const dim_t wind_length,
                  const dim_t wind_width, const af_border_type edge_pad) {
    CHECK_ARRAYS(in);
    CALL(af_maxfilt, out, in, wind_length, wind_width, edge_pad);
}

af_err af_regions(af_array *out, const af_array in,
                  const af_connectivity connectivity, const af_dtype ty) {
    CHECK_ARRAYS(in);
    CALL(af_regions, out, in, connectivity, ty);
}

af_err af_sobel_operator(af_array *dx, af_array *dy, const af_array img,
                         const unsigned ker_size) {
    CHECK_ARRAYS(img);
    CALL(af_sobel_operator, dx, dy, img, ker_size);
}

af_err af_rgb2gray(af_array *out, const af_array in, const float rPercent,
                   const float gPercent, const float bPercent) {
    CHECK_ARRAYS(in);
    CALL(af_rgb2gray, out, in, rPercent, gPercent, bPercent);
}

af_err af_gray2rgb(af_array *out, const af_array in, const float rFactor,
                   const float gFactor, const float bFactor) {
    CHECK_ARRAYS(in);
    CALL(af_gray2rgb, out, in, rFactor, gFactor, bFactor);
}

af_err af_hist_equal(af_array *out, const af_array in, const af_array hist) {
    CHECK_ARRAYS(in, hist);
    CALL(af_hist_equal, out, in, hist);
}

af_err af_gaussian_kernel(af_array *out, const int rows, const int cols,
                          const double sigma_r, const double sigma_c) {
    CALL(af_gaussian_kernel, out, rows, cols, sigma_r, sigma_c);
}

af_err af_hsv2rgb(af_array *out, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_hsv2rgb, out, in);
}

af_err af_rgb2hsv(af_array *out, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_rgb2hsv, out, in);
}

af_err af_color_space(af_array *out, const af_array image, const af_cspace_t to,
                      const af_cspace_t from) {
    CHECK_ARRAYS(image);
    CALL(af_color_space, out, image, to, from);
}

af_err af_unwrap(af_array *out, const af_array in, const dim_t wx,
                 const dim_t wy, const dim_t sx, const dim_t sy, const dim_t px,
                 const dim_t py, const bool is_column) {
    CHECK_ARRAYS(in);
    CALL(af_unwrap, out, in, wx, wy, sx, sy, px, py, is_column);
}

af_err af_wrap(af_array *out, const af_array in, const dim_t ox, const dim_t oy,
               const dim_t wx, const dim_t wy, const dim_t sx, const dim_t sy,
               const dim_t px, const dim_t py, const bool is_column) {
    CHECK_ARRAYS(in);
    CALL(af_wrap, out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);
}

af_err af_wrap_v2(af_array *out, const af_array in, const dim_t ox,
                  const dim_t oy, const dim_t wx, const dim_t wy,
                  const dim_t sx, const dim_t sy, const dim_t px,
                  const dim_t py, const bool is_column) {
    CHECK_ARRAYS(out, in);
    CALL(af_wrap_v2, out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);
}

af_err af_sat(af_array *out, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_sat, out, in);
}

af_err af_ycbcr2rgb(af_array *out, const af_array in,
                    const af_ycc_std standard) {
    CHECK_ARRAYS(in);
    CALL(af_ycbcr2rgb, out, in, standard);
}

af_err af_rgb2ycbcr(af_array *out, const af_array in,
                    const af_ycc_std standard) {
    CHECK_ARRAYS(in);
    CALL(af_rgb2ycbcr, out, in, standard);
}

af_err af_canny(af_array *out, const af_array in, const af_canny_threshold ct,
                const float t1, const float t2, const unsigned sw,
                const bool isf) {
    CHECK_ARRAYS(in);
    CALL(af_canny, out, in, ct, t1, t2, sw, isf);
}

af_err af_anisotropic_diffusion(af_array *out, const af_array in,
                                const float dt, const float K,
                                const unsigned iterations,
                                const af_flux_function fftype,
                                const af_diffusion_eq eq) {
    CHECK_ARRAYS(in);
    CALL(af_anisotropic_diffusion, out, in, dt, K, iterations, fftype, eq);
}

af_err af_iterative_deconv(af_array *out, const af_array in, const af_array ker,
                           const unsigned iterations, const float relax_factor,
                           const af_iterative_deconv_algo algo) {
    CHECK_ARRAYS(in, ker);
    CALL(af_iterative_deconv, out, in, ker, iterations, relax_factor, algo);
}

af_err af_inverse_deconv(af_array *out, const af_array in, const af_array psf,
                         const float gamma, const af_inverse_deconv_algo algo) {
    CHECK_ARRAYS(in, psf);
    CALL(af_inverse_deconv, out, in, psf, gamma, algo);
}

af_err af_confidence_cc(af_array *out, const af_array in, const af_array seedx,
                        const af_array seedy, const unsigned radius,
                        const unsigned multiplier, const int iter,
                        const double segmented_value) {
    CHECK_ARRAYS(in, seedx, seedy);
    CALL(af_confidence_cc, out, in, seedx, seedy, radius, multiplier, iter,
         segmented_value);
}
