/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/graphics.h>
#include "symbol_manager.hpp"


af_err af_create_window(af_window *out, const int width, const int height, const char* const title)
{
    return CALL(out, width, height, title);
}

af_err af_set_position(const af_window wind, const unsigned x, const unsigned y)
{
    return CALL(wind, x, y);
}

af_err af_set_title(const af_window wind, const char* const title)
{
    return CALL(wind, title);
}

af_err af_set_size(const af_window wind, const unsigned w, const unsigned h)
{
    return CALL(wind, w, h);
}

af_err af_draw_image(const af_window wind, const af_array in, const af_cell* const props)
{
    CHECK_ARRAYS(in);
    return CALL(wind, in, props);
}

af_err af_draw_plot(const af_window wind, const af_array X, const af_array Y, const af_cell* const props)
{
    CHECK_ARRAYS(X, Y);
    return CALL(wind, X, Y, props);
}

af_err af_draw_scatter(const af_window wind, const af_array X, const af_array Y, const af_marker_type marker, const af_cell* const props)
{
    CHECK_ARRAYS(X, Y);
    return CALL(wind, X, Y, marker, props);
}

af_err af_draw_scatter3(const af_window wind, const af_array P, const af_marker_type marker, const af_cell* const props)
{
    CHECK_ARRAYS(P);
    return CALL(wind, P, marker, props);
}

af_err af_draw_plot3(const af_window wind, const af_array P, const af_cell* const props)
{
    CHECK_ARRAYS(P);
    return CALL(wind, P, props);
}

af_err af_draw_hist(const af_window wind, const af_array X, const double minval, const double maxval, const af_cell* const props)
{
    CHECK_ARRAYS(X);
    return CALL(wind, X, minval, maxval, props);
}

af_err af_draw_surface(const af_window wind, const af_array xVals, const af_array yVals, const af_array S, const af_cell* const props)
{
    CHECK_ARRAYS(xVals, yVals, S);
    return CALL(wind, xVals, yVals, S, props);
}

af_err af_grid(const af_window wind, const int rows, const int cols)
{
    return CALL(wind, rows, cols);
}

af_err af_show(const af_window wind)
{
    return CALL(wind);
}

af_err af_is_window_closed(bool *out, const af_window wind)
{
    return CALL(out, wind);
}

af_err af_set_visibility(const af_window wind, const bool is_visible)
{
    return CALL(wind, is_visible);
}

af_err af_destroy_window(const af_window wind)
{
    return CALL(wind);
}
