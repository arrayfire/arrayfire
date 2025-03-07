/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/data.h>
#include <af/graphics.h>
#include <af/image.h>

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/graphics_common.hpp>
#include <handle.hpp>
#include <join.hpp>
#include <platform.hpp>
#include <plot.hpp>
#include <reduce.hpp>
#include <reorder.hpp>
#include <transpose.hpp>

using af::dim4;
using arrayfire::common::ForgeManager;
using arrayfire::common::ForgeModule;
using arrayfire::common::forgePlugin;
using arrayfire::common::getFGMarker;
using arrayfire::common::getGLType;
using arrayfire::common::makeContextCurrent;
using arrayfire::common::step_round;
using detail::Array;
using detail::copy_plot;
using detail::forgeManager;
using detail::reduce;
using detail::uchar;
using detail::uint;
using detail::ushort;

// Requires in_ to be in either [order, n] or [n, order] format
template<typename T, int order>
fg_chart setup_plot(fg_window window, const af_array in_,
                    const af_cell* const props, fg_plot_type ptype,
                    fg_marker_type mtype) {
    ForgeModule& _ = forgePlugin();

    Array<T> in = getArray<T>(in_);

    af::dim4 dims = in.dims();

    DIM_ASSERT(1, dims.ndims() == 2);
    DIM_ASSERT(1, (dims[0] == order || dims[1] == order));

    // The data expected by backend is 2D [order, n]
    if (dims[1] == order) { in = transpose(in, false); }

    af::dim4 tdims = in.dims();  // transposed dimensions

    ForgeManager& fgMngr = forgeManager();

    // Get the chart for the current grid position (if any)
    fg_chart chart      = NULL;
    fg_chart_type ctype = order == 2 ? FG_CHART_2D : FG_CHART_3D;

    if (props->col > -1 && props->row > -1) {
        chart = fgMngr.getChart(window, props->row, props->col, ctype);
    } else {
        chart = fgMngr.getChart(window, 0, 0, ctype);
    }

    fg_plot plot =
        fgMngr.getPlot(chart, tdims[1], getGLType<T>(), ptype, mtype);

    // ArrayFire LOGO Orange shade
    FG_CHECK(_.fg_set_plot_color(plot, 0.929f, 0.529f, 0.212f, 1.0));

    // If chart axes limits do not have a manual override
    // then compute and set axes limits
    if (!fgMngr.getChartAxesOverride(chart)) {
        float cmin[3], cmax[3];
        T dmin[3], dmax[3];
        FG_CHECK(_.fg_get_chart_axes_limits(
            &cmin[0], &cmax[0], &cmin[1], &cmax[1], &cmin[2], &cmax[2], chart));
        copyData(dmin, reduce<af_min_t, T, T>(in, 1));
        copyData(dmax, reduce<af_max_t, T, T>(in, 1));

        if (cmin[0] == 0 && cmax[0] == 0 && cmin[1] == 0 && cmax[1] == 0 &&
            cmin[2] == 0 && cmax[2] == 0) {
            // No previous limits. Set without checking
            cmin[0] = step_round(dmin[0], false);
            cmax[0] = step_round(dmax[0], true);
            cmin[1] = step_round(dmin[1], false);
            cmax[1] = step_round(dmax[1], true);
            if (order == 3) { cmin[2] = step_round(dmin[2], false); }
            if (order == 3) { cmax[2] = step_round(dmax[2], true); }
        } else {
            if (cmin[0] > dmin[0]) { cmin[0] = step_round(dmin[0], false); }
            if (cmax[0] < dmax[0]) { cmax[0] = step_round(dmax[0], true); }
            if (cmin[1] > dmin[1]) { cmin[1] = step_round(dmin[1], false); }
            if (cmax[1] < dmax[1]) { cmax[1] = step_round(dmax[1], true); }
            if (order == 3) {
                if (cmin[2] > dmin[2]) { cmin[2] = step_round(dmin[2], false); }
                if (cmax[2] < dmax[2]) { cmax[2] = step_round(dmax[2], true); }
            }
        }
        FG_CHECK(_.fg_set_chart_axes_limits(chart, cmin[0], cmax[0], cmin[1],
                                            cmax[1], cmin[2], cmax[2]));
    }
    copy_plot<T>(in, plot);

    return chart;
}

template<typename T>
fg_chart setup_plot(fg_window window, const af_array in_, const int order,
                    const af_cell* const props, fg_plot_type ptype,
                    fg_marker_type mtype) {
    if (order == 2) {
        return setup_plot<T, 2>(window, in_, props, ptype, mtype);
    }
    if (order == 3) {
        return setup_plot<T, 3>(window, in_, props, ptype, mtype);
    }
    // Dummy to avoid warnings
    return NULL;
}

af_err plotWrapper(const af_window window, const af_array in,
                   const int order_dim, const af_cell* const props,
                   fg_plot_type ptype    = FG_PLOT_LINE,
                   fg_marker_type marker = FG_MARKER_NONE) {
    try {
        if (window == 0) { AF_ERROR("Not a valid window", AF_ERR_INTERNAL); }

        const ArrayInfo& info = getInfo(in);
        af::dim4 dims         = info.dims();
        af_dtype type         = info.getType();

        DIM_ASSERT(0, dims.ndims() == 2);
        DIM_ASSERT(0, dims[order_dim] == 2 || dims[order_dim] == 3);

        makeContextCurrent(window);

        fg_chart chart = NULL;

        switch (type) {
            case f32:
                chart = setup_plot<float>(window, in, dims[order_dim], props,
                                          ptype, marker);
                break;
            case s32:
                chart = setup_plot<int>(window, in, dims[order_dim], props,
                                        ptype, marker);
                break;
            case u32:
                chart = setup_plot<uint>(window, in, dims[order_dim], props,
                                         ptype, marker);
                break;
            case s16:
                chart = setup_plot<short>(window, in, dims[order_dim], props,
                                          ptype, marker);
                break;
            case u16:
                chart = setup_plot<ushort>(window, in, dims[order_dim], props,
                                           ptype, marker);
                break;
            case u8:
                chart = setup_plot<uchar>(window, in, dims[order_dim], props,
                                          ptype, marker);
                break;
            default: TYPE_ERROR(1, type);
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

af_err plotWrapper(const af_window window, const af_array X, const af_array Y,
                   const af_array Z, const af_cell* const props,
                   fg_plot_type ptype    = FG_PLOT_LINE,
                   fg_marker_type marker = FG_MARKER_NONE) {
    try {
        if (window == 0) { AF_ERROR("Not a valid window", AF_ERR_INTERNAL); }

        const ArrayInfo& xInfo = getInfo(X);
        const af::dim4& xDims  = xInfo.dims();
        af_dtype xType         = xInfo.getType();

        const ArrayInfo& yInfo = getInfo(Y);
        const af::dim4& yDims  = yInfo.dims();
        af_dtype yType         = yInfo.getType();

        const ArrayInfo& zInfo = getInfo(Z);
        const af::dim4& zDims  = zInfo.dims();
        af_dtype zType         = zInfo.getType();

        DIM_ASSERT(0, xDims == yDims);
        DIM_ASSERT(0, xDims == zDims);
        DIM_ASSERT(0, xInfo.isVector());

        TYPE_ASSERT(xType == yType);
        TYPE_ASSERT(xType == zType);

        // Join for set up vector
        af_array in    = 0;
        af_array pIn[] = {X, Y, Z};
        AF_CHECK(af_join_many(&in, 1, 3, pIn));

        makeContextCurrent(window);

        fg_chart chart = NULL;

        switch (xType) {
            case f32:
                chart = setup_plot<float>(window, in, 3, props, ptype, marker);
                break;
            case s32:
                chart = setup_plot<int>(window, in, 3, props, ptype, marker);
                break;
            case u32:
                chart = setup_plot<uint>(window, in, 3, props, ptype, marker);
                break;
            case s16:
                chart = setup_plot<short>(window, in, 3, props, ptype, marker);
                break;
            case u16:
                chart = setup_plot<ushort>(window, in, 3, props, ptype, marker);
                break;
            case u8:
                chart = setup_plot<uchar>(window, in, 3, props, ptype, marker);
                break;
            default: TYPE_ERROR(1, xType);
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

        AF_CHECK(af_release_array(in));
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err plotWrapper(const af_window window, const af_array X, const af_array Y,
                   const af_cell* const props,
                   fg_plot_type ptype    = FG_PLOT_LINE,
                   fg_marker_type marker = FG_MARKER_NONE) {
    try {
        if (window == 0) { AF_ERROR("Not a valid window", AF_ERR_INTERNAL); }

        const ArrayInfo& xInfo = getInfo(X);
        const af::dim4& xDims  = xInfo.dims();
        af_dtype xType         = xInfo.getType();

        const ArrayInfo& yInfo = getInfo(Y);
        const af::dim4& yDims  = yInfo.dims();
        af_dtype yType         = yInfo.getType();

        DIM_ASSERT(0, xDims == yDims);
        DIM_ASSERT(0, xInfo.isVector());

        TYPE_ASSERT(xType == yType);

        // Join for set up vector
        af_array in = 0;
        AF_CHECK(af_join(&in, 1, X, Y));

        makeContextCurrent(window);

        fg_chart chart = NULL;

        switch (xType) {
            case f32:
                chart = setup_plot<float>(window, in, 2, props, ptype, marker);
                break;
            case s32:
                chart = setup_plot<int>(window, in, 2, props, ptype, marker);
                break;
            case u32:
                chart = setup_plot<uint>(window, in, 2, props, ptype, marker);
                break;
            case s16:
                chart = setup_plot<short>(window, in, 2, props, ptype, marker);
                break;
            case u16:
                chart = setup_plot<ushort>(window, in, 2, props, ptype, marker);
                break;
            case u8:
                chart = setup_plot<uchar>(window, in, 2, props, ptype, marker);
                break;
            default: TYPE_ERROR(1, xType);
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

        AF_CHECK(af_release_array(in));
    }
    CATCHALL;
    return AF_SUCCESS;
}

// Plot API
af_err af_draw_plot_nd(const af_window wind, const af_array in,
                       const af_cell* const props) {
    return plotWrapper(wind, in, 1, props);
}

af_err af_draw_plot_2d(const af_window wind, const af_array X, const af_array Y,
                       const af_cell* const props) {
    return plotWrapper(wind, X, Y, props);
}

af_err af_draw_plot_3d(const af_window wind, const af_array X, const af_array Y,
                       const af_array Z, const af_cell* const props) {
    return plotWrapper(wind, X, Y, Z, props);
}

// Deprecated Plot API
af_err af_draw_plot(const af_window wind, const af_array X, const af_array Y,
                    const af_cell* const props) {
    return plotWrapper(wind, X, Y, props);
}

af_err af_draw_plot3(const af_window wind, const af_array P,
                     const af_cell* const props) {
    try {
        const ArrayInfo& info = getInfo(P);
        af::dim4 dims         = info.dims();

        if (dims.ndims() == 2 && dims[1] == 3) {
            return plotWrapper(wind, P, 1, props);
        }
        if (dims.ndims() == 2 && dims[0] == 3) {
            return plotWrapper(wind, P, 0, props);
        } else if (dims.ndims() == 1 && dims[0] % 3 == 0) {
            dim4 rdims(dims.elements() / 3, 3, 1, 1);
            af_array in = 0;
            AF_CHECK(af_moddims(&in, P, rdims.ndims(), rdims.get()));
            af_err err = plotWrapper(wind, in, 1, props);
            AF_CHECK(af_release_array(in));
            return err;
        } else {
            AF_RETURN_ERROR(
                "Input needs to be either [n, 3] or [3, n] or [3n, 1]",
                AF_ERR_SIZE);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

// Scatter API
af_err af_draw_scatter_nd(const af_window wind, const af_array in,
                          const af_marker_type af_marker,
                          const af_cell* const props) {
    try {
        fg_marker_type fg_marker = getFGMarker(af_marker);
        return plotWrapper(wind, in, 1, props, FG_PLOT_SCATTER, fg_marker);
    }
    CATCHALL;
}

af_err af_draw_scatter_2d(const af_window wind, const af_array X,
                          const af_array Y, const af_marker_type af_marker,
                          const af_cell* const props) {
    try {
        fg_marker_type fg_marker = getFGMarker(af_marker);
        return plotWrapper(wind, X, Y, props, FG_PLOT_SCATTER, fg_marker);
    }
    CATCHALL;
}

af_err af_draw_scatter_3d(const af_window wind, const af_array X,
                          const af_array Y, const af_array Z,
                          const af_marker_type af_marker,
                          const af_cell* const props) {
    try {
        fg_marker_type fg_marker = getFGMarker(af_marker);
        return plotWrapper(wind, X, Y, Z, props, FG_PLOT_SCATTER, fg_marker);
    }
    CATCHALL;
}

// Deprecated Scatter API
af_err af_draw_scatter(const af_window wind, const af_array X, const af_array Y,
                       const af_marker_type af_marker,
                       const af_cell* const props) {
    try {
        fg_marker_type fg_marker = getFGMarker(af_marker);
        return plotWrapper(wind, X, Y, props, FG_PLOT_SCATTER, fg_marker);
    }
    CATCHALL;
}

af_err af_draw_scatter3(const af_window wind, const af_array P,
                        const af_marker_type af_marker,
                        const af_cell* const props) {
    try {
        fg_marker_type fg_marker = getFGMarker(af_marker);
        const ArrayInfo& info    = getInfo(P);
        af::dim4 dims            = info.dims();

        if (dims.ndims() == 2 && dims[1] == 3) {
            return plotWrapper(wind, P, 1, props, FG_PLOT_SCATTER, fg_marker);
        }
        if (dims.ndims() == 2 && dims[0] == 3) {
            return plotWrapper(wind, P, 0, props, FG_PLOT_SCATTER, fg_marker);
        } else if (dims.ndims() == 1 && dims[0] % 3 == 0) {
            dim4 rdims(dims.elements() / 3, 3, 1, 1);
            af_array in = 0;
            AF_CHECK(af_moddims(&in, P, rdims.ndims(), rdims.get()));
            af_err err =
                plotWrapper(wind, in, 1, props, FG_PLOT_SCATTER, fg_marker);
            AF_CHECK(af_release_array(in));
            return err;
        } else {
            AF_RETURN_ERROR(
                "Input needs to be either [n, 3] or [3, n] or [3n, 1]",
                AF_ERR_SIZE);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}
