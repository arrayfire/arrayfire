/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/deprecated.hpp>
#include <af/array.h>
#include <af/graphics.h>
#include "symbol_manager.hpp"

af_err af_create_window(af_window* out, const int width, const int height,
                        const char* const title) {
    CALL(af_create_window, out, width, height, title);
}

af_err af_set_position(const af_window wind, const unsigned x,
                       const unsigned y) {
    CALL(af_set_position, wind, x, y);
}

af_err af_set_title(const af_window wind, const char* const title) {
    CALL(af_set_title, wind, title);
}

af_err af_set_size(const af_window wind, const unsigned w, const unsigned h) {
    CALL(af_set_size, wind, w, h);
}

af_err af_draw_image(const af_window wind, const af_array in,
                     const af_cell* const props) {
    CHECK_ARRAYS(in);
    CALL(af_draw_image, wind, in, props);
}

af_err af_draw_plot(const af_window wind, const af_array X, const af_array Y,
                    const af_cell* const props) {
    CHECK_ARRAYS(X, Y);
    AF_DEPRECATED_WARNINGS_OFF
    CALL(af_draw_plot, wind, X, Y, props);
    AF_DEPRECATED_WARNINGS_ON
}

af_err af_draw_plot3(const af_window wind, const af_array P,
                     const af_cell* const props) {
    CHECK_ARRAYS(P);
    AF_DEPRECATED_WARNINGS_OFF
    CALL(af_draw_plot3, wind, P, props);
    AF_DEPRECATED_WARNINGS_ON
}

af_err af_draw_plot_nd(const af_window wind, const af_array in,
                       const af_cell* const props) {
    CHECK_ARRAYS(in);
    CALL(af_draw_plot_nd, wind, in, props);
}

af_err af_draw_plot_2d(const af_window wind, const af_array X, const af_array Y,
                       const af_cell* const props) {
    CHECK_ARRAYS(X, Y);
    CALL(af_draw_plot_2d, wind, X, Y, props);
}

af_err af_draw_plot_3d(const af_window wind, const af_array X, const af_array Y,
                       const af_array Z, const af_cell* const props) {
    CHECK_ARRAYS(X, Y, Z);
    CALL(af_draw_plot_3d, wind, X, Y, Z, props);
}

af_err af_draw_scatter(const af_window wind, const af_array X, const af_array Y,
                       const af_marker_type marker,
                       const af_cell* const props) {
    CHECK_ARRAYS(X, Y);
    AF_DEPRECATED_WARNINGS_OFF
    CALL(af_draw_scatter, wind, X, Y, marker, props);
    AF_DEPRECATED_WARNINGS_ON
}

af_err af_draw_scatter3(const af_window wind, const af_array P,
                        const af_marker_type marker,
                        const af_cell* const props) {
    CHECK_ARRAYS(P);
    AF_DEPRECATED_WARNINGS_OFF
    CALL(af_draw_scatter3, wind, P, marker, props);
    AF_DEPRECATED_WARNINGS_ON
}

af_err af_draw_scatter_nd(const af_window wind, const af_array in,
                          const af_marker_type marker,
                          const af_cell* const props) {
    CHECK_ARRAYS(in);
    CALL(af_draw_scatter_nd, wind, in, marker, props);
}

af_err af_draw_scatter_2d(const af_window wind, const af_array X,
                          const af_array Y, const af_marker_type marker,
                          const af_cell* const props) {
    CHECK_ARRAYS(X, Y);
    CALL(af_draw_scatter_2d, wind, X, Y, marker, props);
}

af_err af_draw_scatter_3d(const af_window wind, const af_array X,
                          const af_array Y, const af_array Z,
                          const af_marker_type marker,
                          const af_cell* const props) {
    CHECK_ARRAYS(X, Y, Z);
    CALL(af_draw_scatter_3d, wind, X, Y, Z, marker, props);
}

af_err af_draw_hist(const af_window wind, const af_array X, const double minval,
                    const double maxval, const af_cell* const props) {
    CHECK_ARRAYS(X);
    CALL(af_draw_hist, wind, X, minval, maxval, props);
}

af_err af_draw_surface(const af_window wind, const af_array xVals,
                       const af_array yVals, const af_array S,
                       const af_cell* const props) {
    CHECK_ARRAYS(xVals, yVals, S);
    CALL(af_draw_surface, wind, xVals, yVals, S, props);
}

af_err af_draw_vector_field_nd(const af_window wind, const af_array points,
                               const af_array directions,
                               const af_cell* const props) {
    CHECK_ARRAYS(points, directions);
    CALL(af_draw_vector_field_nd, wind, points, directions, props);
}

af_err af_draw_vector_field_3d(const af_window wind, const af_array xPoints,
                               const af_array yPoints, const af_array zPoints,
                               const af_array xDirs, const af_array yDirs,
                               const af_array zDirs,
                               const af_cell* const props) {
    CHECK_ARRAYS(xPoints, yPoints, zPoints, xDirs, yDirs, zDirs);
    CALL(af_draw_vector_field_3d, wind, xPoints, yPoints, zPoints, xDirs, yDirs,
         zDirs, props);
}

af_err af_draw_vector_field_2d(const af_window wind, const af_array xPoints,
                               const af_array yPoints, const af_array xDirs,
                               const af_array yDirs,
                               const af_cell* const props) {
    CHECK_ARRAYS(xPoints, yPoints, xDirs, yDirs);
    CALL(af_draw_vector_field_2d, wind, xPoints, yPoints, xDirs, yDirs, props);
}

af_err af_grid(const af_window wind, const int rows, const int cols) {
    CALL(af_grid, wind, rows, cols);
}

af_err af_set_axes_limits_compute(const af_window wind, const af_array x,
                                  const af_array y, const af_array z,
                                  const bool exact,
                                  const af_cell* const props) {
    CHECK_ARRAYS(x, y);
    if (z) { CHECK_ARRAYS(z); }
    CALL(af_set_axes_limits_compute, wind, x, y, z, exact, props);
}

af_err af_set_axes_limits_2d(const af_window wind, const float xmin,
                             const float xmax, const float ymin,
                             const float ymax, const bool exact,
                             const af_cell* const props) {
    CALL(af_set_axes_limits_2d, wind, xmin, xmax, ymin, ymax, exact, props);
}

af_err af_set_axes_limits_3d(const af_window wind, const float xmin,
                             const float xmax, const float ymin,
                             const float ymax, const float zmin,
                             const float zmax, const bool exact,
                             const af_cell* const props) {
    CALL(af_set_axes_limits_3d, wind, xmin, xmax, ymin, ymax, zmin, zmax, exact,
         props);
}

af_err af_set_axes_titles(const af_window wind, const char* const xtitle,
                          const char* const ytitle, const char* const ztitle,
                          const af_cell* const props) {
    CALL(af_set_axes_titles, wind, xtitle, ytitle, ztitle, props);
}

af_err af_set_axes_label_format(const af_window wind, const char* const xformat,
                                const char* const yformat,
                                const char* const zformat,
                                const af_cell* const props) {
    CALL(af_set_axes_label_format, wind, xformat, yformat, zformat, props);
}

af_err af_show(const af_window wind) { CALL(af_show, wind); }

af_err af_is_window_closed(bool* out, const af_window wind) {
    CALL(af_is_window_closed, out, wind);
}

af_err af_set_visibility(const af_window wind, const bool is_visible) {
    CALL(af_set_visibility, wind, is_visible);
}

af_err af_destroy_window(const af_window wind) {
    CALL(af_destroy_window, wind);
}
