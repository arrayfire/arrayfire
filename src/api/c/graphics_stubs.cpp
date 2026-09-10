/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Compiled instead of the graphics sources when AF_WITH_GRAPHICS is OFF.
// Every graphics entry point reports AF_ERR_NOT_CONFIGURED, the same way
// the image IO functions do when FreeImage is not built in.

#include <common/err_common.hpp>
#include <af/graphics.h>

#define AF_GRAPHICS_NOT_CONFIGURED()                                        \
    AF_RETURN_ERROR(                                                     \
        "ArrayFire compiled without graphics support (AF_WITH_GRAPHICS)", \
        AF_ERR_NOT_CONFIGURED)

af_err af_create_window(af_window *, const int, const int, const char* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_set_position(const af_window, const unsigned, const unsigned) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_set_title(const af_window, const char* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_set_size(const af_window, const unsigned, const unsigned) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_draw_image(const af_window, const af_array, const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_draw_plot(const af_window, const af_array, const af_array,
                    const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_draw_plot3(const af_window, const af_array, const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_draw_plot_nd(const af_window, const af_array, const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_draw_plot_2d(const af_window, const af_array, const af_array,
                       const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_draw_plot_3d(const af_window, const af_array, const af_array,
                       const af_array, const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_draw_scatter(const af_window, const af_array, const af_array,
                       const af_marker_type, const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_draw_scatter3(const af_window, const af_array, const af_marker_type,
                        const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_draw_scatter_nd(const af_window, const af_array,
                          const af_marker_type, const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_draw_scatter_2d(const af_window, const af_array, const af_array,
                          const af_marker_type, const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_draw_scatter_3d(const af_window, const af_array, const af_array,
                          const af_array, const af_marker_type,
                          const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_draw_hist(const af_window, const af_array, const double,
                    const double, const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_draw_surface(const af_window, const af_array, const af_array,
                       const af_array, const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_draw_vector_field_nd(const af_window, const af_array,
                               const af_array, const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_draw_vector_field_3d(const af_window, const af_array,
                               const af_array, const af_array, const af_array,
                               const af_array, const af_array,
                               const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_draw_vector_field_2d(const af_window, const af_array,
                               const af_array, const af_array, const af_array,
                               const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_grid(const af_window, const int, const int) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_set_axes_limits_compute(const af_window, const af_array,
                                  const af_array, const af_array, const bool,
                                  const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_set_axes_limits_2d(const af_window, const float, const float,
                             const float, const float, const bool,
                             const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_set_axes_limits_3d(const af_window, const float, const float,
                             const float, const float, const float,
                             const float, const bool, const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_set_axes_titles(const af_window, const char * const,
                          const char * const, const char * const,
                          const af_cell* const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_set_axes_label_format(const af_window, const char *const,
                                const char *const, const char *const,
                                const af_cell *const) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_show(const af_window) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_is_window_closed(bool *, const af_window) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_set_visibility(const af_window, const bool) {
    AF_GRAPHICS_NOT_CONFIGURED();
}

af_err af_destroy_window(const af_window) {
    AF_GRAPHICS_NOT_CONFIGURED();
}
