/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/graphics.h>
#include <af/image.h>

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/graphics_common.hpp>
#include <common/moddims.hpp>
#include <common/tile.hpp>
#include <copy.hpp>
#include <handle.hpp>
#include <join.hpp>
#include <reduce.hpp>
#include <reorder.hpp>
#include <surface.hpp>

using af::dim4;
using arrayfire::common::ForgeManager;
using arrayfire::common::ForgeModule;
using arrayfire::common::forgePlugin;
using arrayfire::common::getGLType;
using arrayfire::common::makeContextCurrent;
using arrayfire::common::modDims;
using arrayfire::common::step_round;
using detail::Array;
using detail::copy_surface;
using detail::createEmptyArray;
using detail::forgeManager;
using detail::reduce_all;
using detail::uchar;
using detail::uint;
using detail::ushort;

template<typename T>
fg_chart setup_surface(fg_window window, const af_array xVals,
                       const af_array yVals, const af_array zVals,
                       const af_cell* const props) {
    ForgeModule& _ = forgePlugin();
    Array<T> xIn   = getArray<T>(xVals);
    Array<T> yIn   = getArray<T>(yVals);
    Array<T> zIn   = getArray<T>(zVals);

    const ArrayInfo& Xinfo = getInfo(xVals);
    const ArrayInfo& Yinfo = getInfo(yVals);
    const ArrayInfo& Zinfo = getInfo(zVals);

    dim4 X_dims = Xinfo.dims();
    dim4 Y_dims = Yinfo.dims();
    dim4 Z_dims = Zinfo.dims();

    if (Xinfo.isVector()) {
        // Convert xIn is a column vector
        xIn = modDims(xIn, xIn.elements());
        // Now tile along second dimension
        dim4 x_tdims(1, Y_dims[0], 1, 1);
        xIn = arrayfire::common::tile(xIn, x_tdims);

        // Convert yIn to a row vector
        yIn = modDims(yIn, dim4(1, yIn.elements()));
        // Now tile along first dimension
        dim4 y_tdims(X_dims[0], 1, 1, 1);
        yIn = arrayfire::common::tile(yIn, y_tdims);
    }

    // Flatten xIn, yIn and zIn into row vectors
    dim4 rowDims = dim4(1, zIn.elements());
    xIn          = modDims(xIn, rowDims);
    yIn          = modDims(yIn, rowDims);
    zIn          = modDims(zIn, rowDims);

    // Now join along first dimension, skip reorder
    std::vector<Array<T>> inputs{xIn, yIn, zIn};

    dim4 odims(3, rowDims[1]);
    Array<T> out = createEmptyArray<T>(odims);
    join(out, 0, inputs);
    Array<T> Z = out;

    ForgeManager& fgMngr = forgeManager();

    // Get the chart for the current grid position (if any)
    fg_chart chart = NULL;
    if (props->col > -1 && props->row > -1) {
        chart = fgMngr.getChart(window, props->row, props->col, FG_CHART_3D);
    } else {
        chart = fgMngr.getChart(window, 0, 0, FG_CHART_3D);
    }

    fg_surface surface =
        fgMngr.getSurface(chart, Z_dims[0], Z_dims[1], getGLType<T>());

    FG_CHECK(_.fg_set_surface_color(surface, 0.0, 1.0, 0.0, 1.0));

    // If chart axes limits do not have a manual override
    // then compute and set axes limits
    if (!fgMngr.getChartAxesOverride(chart)) {
        float cmin[3], cmax[3];
        T dmin[3], dmax[3];
        FG_CHECK(_.fg_get_chart_axes_limits(
            &cmin[0], &cmax[0], &cmin[1], &cmax[1], &cmin[2], &cmax[2], chart));
        dmin[0] = reduce_all<af_min_t, T, T>(xIn);
        dmax[0] = reduce_all<af_max_t, T, T>(xIn);
        dmin[1] = reduce_all<af_min_t, T, T>(yIn);
        dmax[1] = reduce_all<af_max_t, T, T>(yIn);
        dmin[2] = reduce_all<af_min_t, T, T>(zIn);
        dmax[2] = reduce_all<af_max_t, T, T>(zIn);

        if (cmin[0] == 0 && cmax[0] == 0 && cmin[1] == 0 && cmax[1] == 0 &&
            cmin[2] == 0 && cmax[2] == 0) {
            // No previous limits. Set without checking
            cmin[0] = step_round(dmin[0], false);
            cmax[0] = step_round(dmax[0], true);
            cmin[1] = step_round(dmin[1], false);
            cmax[1] = step_round(dmax[1], true);
            cmin[2] = step_round(dmin[2], false);
            cmax[2] = step_round(dmax[2], true);
        } else {
            if (cmin[0] > dmin[0]) { cmin[0] = step_round(dmin[0], false); }
            if (cmax[0] < dmax[0]) { cmax[0] = step_round(dmax[0], true); }
            if (cmin[1] > dmin[1]) { cmin[1] = step_round(dmin[1], false); }
            if (cmax[1] < dmax[1]) { cmax[1] = step_round(dmax[1], true); }
            if (cmin[2] > dmin[2]) { cmin[2] = step_round(dmin[2], false); }
            if (cmax[2] < dmax[2]) { cmax[2] = step_round(dmax[2], true); }
        }

        FG_CHECK(_.fg_set_chart_axes_limits(chart, cmin[0], cmax[0], cmin[1],
                                            cmax[1], cmin[2], cmax[2]));
    }
    copy_surface<T>(Z, surface);

    return chart;
}

af_err af_draw_surface(const af_window window, const af_array xVals,
                       const af_array yVals, const af_array S,
                       const af_cell* const props) {
    try {
        if (window == 0) { AF_ERROR("Not a valid window", AF_ERR_INTERNAL); }

        const ArrayInfo& Xinfo = getInfo(xVals);
        dim4 X_dims            = Xinfo.dims();
        af_dtype Xtype         = Xinfo.getType();

        const ArrayInfo& Yinfo = getInfo(yVals);
        dim4 Y_dims            = Yinfo.dims();
        af_dtype Ytype         = Yinfo.getType();

        const ArrayInfo& Sinfo = getInfo(S);
        const dim4& S_dims     = Sinfo.dims();
        af_dtype Stype         = Sinfo.getType();

        TYPE_ASSERT(Xtype == Ytype);
        TYPE_ASSERT(Ytype == Stype);

        if (!Yinfo.isVector()) {
            DIM_ASSERT(1, X_dims == Y_dims);
            DIM_ASSERT(3, Y_dims == S_dims);
        } else {
            DIM_ASSERT(3, (X_dims[0] * Y_dims[0] == (dim_t)Sinfo.elements()));
        }

        makeContextCurrent(window);

        fg_chart chart = NULL;

        switch (Xtype) {
            case f32:
                chart = setup_surface<float>(window, xVals, yVals, S, props);
                break;
            case s32:
                chart = setup_surface<int>(window, xVals, yVals, S, props);
                break;
            case u32:
                chart = setup_surface<uint>(window, xVals, yVals, S, props);
                break;
            case s16:
                chart = setup_surface<short>(window, xVals, yVals, S, props);
                break;
            case u16:
                chart = setup_surface<ushort>(window, xVals, yVals, S, props);
                break;
            case u8:
                chart = setup_surface<uchar>(window, xVals, yVals, S, props);
                break;
            default: TYPE_ERROR(1, Xtype);
        }
        auto gridDims = forgeManager().getWindowGrid(window);

        ForgeModule& _ = forgePlugin();
        if (props->col > -1 && props->row > -1) {
            FG_CHECK(_.fg_draw_chart_to_cell(
                window, gridDims.first, gridDims.second,
                props->row * gridDims.second + props->col, chart,
                props->title));
        } else {
            FG_CHECK(_.fg_draw_chart(window, chart));
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}
