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

#ifdef __cplusplus
#include <utility>
namespace af
{
    AFAPI void image(const array &in);

    AFAPI void plot(const array &X, const array &Y);

    AFAPI void hist(const array &X, const double minval, const double maxval);

    AFAPI void setupGrid(int rows, int cols);

    AFAPI void bindCell(int colId, int rowId);

    AFAPI void showGrid();
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    AFAPI af_err af_draw_image(const af_array in);

    AFAPI af_err af_draw_plot(const af_array X, const af_array Y);

    AFAPI af_err af_draw_hist(const af_array X, const double minval, const double maxval);

    AFAPI af_err af_setup_grid(int rows, int cols);

    AFAPI af_err af_bind_cell(int colId, int rowId);

    AFAPI af_err af_show_grid();
#ifdef __cplusplus
}

#endif
