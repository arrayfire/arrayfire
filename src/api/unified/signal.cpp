/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/dim4.hpp>
#include <af/signal.h>
#include "symbol_manager.hpp"

af_err af_approx1(af_array *yo, const af_array yi, const af_array xo,
                  const af_interp_type method, const float offGrid) {
    CHECK_ARRAYS(yo, yi, xo);
    CALL(af_approx1, yo, yi, xo, method, offGrid);
}

af_err af_approx1_v2(af_array *yo, const af_array yi, const af_array xo,
                     const af_interp_type method, const float offGrid) {
    CHECK_ARRAYS(yo, yi, xo);
    CALL(af_approx1_v2, yo, yi, xo, method, offGrid);
}

af_err af_approx2(af_array *zo, const af_array zi, const af_array xo,
                  const af_array yo, const af_interp_type method,
                  const float offGrid) {
    CHECK_ARRAYS(zo, zi, xo, yo);
    CALL(af_approx2, zo, zi, xo, yo, method, offGrid);
}

af_err af_approx2_v2(af_array *zo, const af_array zi, const af_array xo,
                     const af_array yo, const af_interp_type method,
                     const float offGrid) {
    CHECK_ARRAYS(zo, zi, xo, yo);
    CALL(af_approx2_v2, zo, zi, xo, yo, method, offGrid);
}

af_err af_approx1_uniform(af_array *yo, const af_array yi, const af_array xo,
                          const int xdim, const double xi_beg,
                          const double xi_step, const af_interp_type method,
                          const float offGrid) {
    CHECK_ARRAYS(yo, yi, xo);
    CALL(af_approx1_uniform, yo, yi, xo, xdim, xi_beg, xi_step, method,
         offGrid);
}

af_err af_approx1_uniform_v2(af_array *yo, const af_array yi, const af_array xo,
                             const int xdim, const double xi_beg,
                             const double xi_step, const af_interp_type method,
                             const float offGrid) {
    CHECK_ARRAYS(yo, yi, xo);
    CALL(af_approx1_uniform_v2, yo, yi, xo, xdim, xi_beg, xi_step, method,
         offGrid);
}

af_err af_approx2_uniform(af_array *zo, const af_array zi, const af_array xo,
                          const int xdim, const double xi_beg,
                          const double xi_step, const af_array yo,
                          const int ydim, const double yi_beg,
                          const double yi_step, const af_interp_type method,
                          const float offGrid) {
    CHECK_ARRAYS(zo, zi, xo, yo);
    CALL(af_approx2_uniform, zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim,
         yi_beg, yi_step, method, offGrid);
}

af_err af_approx2_uniform_v2(af_array *zo, const af_array zi, const af_array xo,
                             const int xdim, const double xi_beg,
                             const double xi_step, const af_array yo,
                             const int ydim, const double yi_beg,
                             const double yi_step, const af_interp_type method,
                             const float offGrid) {
    CHECK_ARRAYS(zo, zi, xo, yo);
    CALL(af_approx2_uniform_v2, zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim,
         yi_beg, yi_step, method, offGrid);
}

af_err af_set_fft_plan_cache_size(size_t cache_size) {
    CALL(af_set_fft_plan_cache_size, cache_size);
}

#define FFT_HAPI_DEF(af_func)                               \
    af_err af_func(af_array in, const double norm_factor) { \
        CHECK_ARRAYS(in);                                   \
        CALL(af_func, in, norm_factor);                     \
    }

FFT_HAPI_DEF(af_fft_inplace)
FFT_HAPI_DEF(af_fft2_inplace)
FFT_HAPI_DEF(af_fft3_inplace)
FFT_HAPI_DEF(af_ifft_inplace)
FFT_HAPI_DEF(af_ifft2_inplace)
FFT_HAPI_DEF(af_ifft3_inplace)

af_err af_fft(af_array *out, const af_array in, const double norm_factor,
              const dim_t odim0) {
    CHECK_ARRAYS(in);
    CALL(af_fft, out, in, norm_factor, odim0);
}

af_err af_fft2(af_array *out, const af_array in, const double norm_factor,
               const dim_t odim0, const dim_t odim1) {
    CHECK_ARRAYS(in);
    CALL(af_fft2, out, in, norm_factor, odim0, odim1);
}

af_err af_fft3(af_array *out, const af_array in, const double norm_factor,
               const dim_t odim0, const dim_t odim1, const dim_t odim2) {
    CHECK_ARRAYS(in);
    CALL(af_fft3, out, in, norm_factor, odim0, odim1, odim2);
}

af_err af_ifft(af_array *out, const af_array in, const double norm_factor,
               const dim_t odim0) {
    CHECK_ARRAYS(in);
    CALL(af_ifft, out, in, norm_factor, odim0);
}

af_err af_ifft2(af_array *out, const af_array in, const double norm_factor,
                const dim_t odim0, const dim_t odim1) {
    CHECK_ARRAYS(in);
    CALL(af_ifft2, out, in, norm_factor, odim0, odim1);
}

af_err af_ifft3(af_array *out, const af_array in, const double norm_factor,
                const dim_t odim0, const dim_t odim1, const dim_t odim2) {
    CHECK_ARRAYS(in);
    CALL(af_ifft3, out, in, norm_factor, odim0, odim1, odim2);
}

af_err af_fft_r2c(af_array *out, const af_array in, const double norm_factor,
                  const dim_t pad0) {
    CHECK_ARRAYS(in);
    CALL(af_fft_r2c, out, in, norm_factor, pad0);
}

af_err af_fft2_r2c(af_array *out, const af_array in, const double norm_factor,
                   const dim_t pad0, const dim_t pad1) {
    CHECK_ARRAYS(in);
    CALL(af_fft2_r2c, out, in, norm_factor, pad0, pad1);
}

af_err af_fft3_r2c(af_array *out, const af_array in, const double norm_factor,
                   const dim_t pad0, const dim_t pad1, const dim_t pad2) {
    CHECK_ARRAYS(in);
    CALL(af_fft3_r2c, out, in, norm_factor, pad0, pad1, pad2);
}

#define FFTC2R_HAPI_DEF(af_func)                                               \
    af_err af_func(af_array *out, const af_array in, const double norm_factor, \
                   const bool is_odd) {                                        \
        CHECK_ARRAYS(in);                                                      \
        CALL(af_func, out, in, norm_factor, is_odd);                           \
    }

FFTC2R_HAPI_DEF(af_fft_c2r)
FFTC2R_HAPI_DEF(af_fft2_c2r)
FFTC2R_HAPI_DEF(af_fft3_c2r)

#define CONV_HAPI_DEF(af_func)                                     \
    af_err af_func(af_array *out, const af_array signal,           \
                   const af_array filter, const af_conv_mode mode, \
                   af_conv_domain domain) {                        \
        CHECK_ARRAYS(signal, filter);                              \
        CALL(af_func, out, signal, filter, mode, domain);          \
    }

CONV_HAPI_DEF(af_convolve1)
CONV_HAPI_DEF(af_convolve2)
CONV_HAPI_DEF(af_convolve3)

af_err af_convolve2_nn(af_array *out, const af_array signal,
                       const af_array filter, const unsigned stride_dims,
                       const dim_t *strides, const unsigned padding_dims,
                       const dim_t *paddings, const unsigned dilation_dims,
                       const dim_t *dilations) {
    CHECK_ARRAYS(signal, filter);
    CALL(af_convolve2_nn, out, signal, filter, stride_dims, strides,
         padding_dims, paddings, dilation_dims, dilations);
}

af_err af_convolve2_gradient_nn(
    af_array *out, const af_array incoming_gradient,
    const af_array original_signal, const af_array original_filter,
    const af_array convolved_output, const unsigned stride_dims,
    const dim_t *strides, const unsigned padding_dims, const dim_t *paddings,
    const unsigned dilation_dims, const dim_t *dilations,
    af_conv_gradient_type grad_type) {
    CHECK_ARRAYS(incoming_gradient, original_signal, original_filter,
                 convolved_output);
    CALL(af_convolve2_gradient_nn, out, incoming_gradient, original_signal,
         original_filter, convolved_output, stride_dims, strides, padding_dims,
         paddings, dilation_dims, dilations, grad_type);
}

#define FFT_CONV_HAPI_DEF(af_func)                                   \
    af_err af_func(af_array *out, const af_array signal,             \
                   const af_array filter, const af_conv_mode mode) { \
        CHECK_ARRAYS(signal, filter);                                \
        CALL(af_func, out, signal, filter, mode);                    \
    }

FFT_CONV_HAPI_DEF(af_fft_convolve1)
FFT_CONV_HAPI_DEF(af_fft_convolve2)
FFT_CONV_HAPI_DEF(af_fft_convolve3)

af_err af_convolve2_sep(af_array *out, const af_array col_filter,
                        const af_array row_filter, const af_array signal,
                        const af_conv_mode mode) {
    CHECK_ARRAYS(col_filter, row_filter, signal);
    CALL(af_convolve2_sep, out, col_filter, row_filter, signal, mode);
}

af_err af_fir(af_array *y, const af_array b, const af_array x) {
    CHECK_ARRAYS(b, x);
    CALL(af_fir, y, b, x);
}

af_err af_iir(af_array *y, const af_array b, const af_array a,
              const af_array x) {
    CHECK_ARRAYS(b, a, x);
    CALL(af_iir, y, b, a, x);
}

af_err af_medfilt(af_array *out, const af_array in, const dim_t wind_length,
                  const dim_t wind_width, const af_border_type edge_pad) {
    CHECK_ARRAYS(in);
    CALL(af_medfilt, out, in, wind_length, wind_width, edge_pad);
}

af_err af_medfilt1(af_array *out, const af_array in, const dim_t wind_width,
                   const af_border_type edge_pad) {
    CHECK_ARRAYS(in);
    CALL(af_medfilt1, out, in, wind_width, edge_pad);
}

af_err af_medfilt2(af_array *out, const af_array in, const dim_t wind_length,
                   const dim_t wind_width, const af_border_type edge_pad) {
    CHECK_ARRAYS(in);
    CALL(af_medfilt2, out, in, wind_length, wind_width, edge_pad);
}
