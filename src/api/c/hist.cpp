/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <common/graphics_common.hpp>
#include <copy.hpp>
#include <handle.hpp>
#include <hist_graphics.hpp>
#include <platform.hpp>
#include <reduce.hpp>
#include <af/graphics.h>

using arrayfire::common::ForgeManager;
using arrayfire::common::ForgeModule;
using arrayfire::common::forgePlugin;
using arrayfire::common::getGLType;
using arrayfire::common::makeContextCurrent;
using arrayfire::common::step_round;
using detail::Array;
using detail::copy_histogram;
using detail::forgeManager;
using detail::getScalar;
using detail::uchar;
using detail::uint;
using detail::ushort;

template<typename T>
fg_chart setup_histogram(fg_window const window, const af_array in,
                         const double minval, const double maxval,
                         const af_cell* const props) {
    ForgeModule& _ = forgePlugin();

    const Array<T> histogramInput = getArray<T>(in);
    dim_t nBins                   = histogramInput.elements();

    // Retrieve Forge Histogram with nBins and array type
    ForgeManager& fgMngr = forgeManager();

    // Get the chart for the current grid position (if any)
    fg_chart chart = NULL;
    if (props->col > -1 && props->row > -1) {
        chart = fgMngr.getChart(window, props->row, props->col, FG_CHART_2D);
    } else {
        chart = fgMngr.getChart(window, 0, 0, FG_CHART_2D);
    }

    // Create a histogram for the chart
    fg_histogram hist = fgMngr.getHistogram(chart, nBins, getGLType<T>());

    // Set histogram bar colors to ArrayFire's orange
    FG_CHECK(_.fg_set_histogram_color(hist, 0.929f, 0.486f, 0.2745f, 1.0f));

    // If chart axes limits do not have a manual override
    // then compute and set axes limits
    if (!fgMngr.getChartAxesOverride(chart)) {
        float xMin, xMax, yMin, yMax, zMin, zMax;
        FG_CHECK(_.fg_get_chart_axes_limits(&xMin, &xMax, &yMin, &yMax, &zMin,
                                            &zMax, chart));
        T freqMax =
            getScalar<T>(detail::reduce_all<af_max_t, T, T>(histogramInput));

	// For histogram, xMin and xMax should always be the first
	// and last bin respectively and should not be rounded
        if (xMin == 0 && xMax == 0 && yMin == 0 && yMax == 0) {
            // No previous limits. Set without checking
            xMin = static_cast<float>(minval);
            xMax = static_cast<float>(maxval);
            yMax = static_cast<float>(step_round(freqMax, true));
            // For histogram, always set yMin to 0.
            yMin = 0;
        } else {
            if (xMin > minval) {
                xMin = static_cast<float>(minval);
            }
            if (xMax < maxval) {
                xMax = static_cast<float>(maxval);
            }
            if (yMax < freqMax) {
                yMax = static_cast<float>(step_round(freqMax, true));
            }
            // For histogram, always set yMin to 0.
            yMin = 0;
        }
        FG_CHECK(_.fg_set_chart_axes_limits(chart, xMin, xMax, yMin, yMax, zMin,
                                            zMax));
    }

    copy_histogram<T>(histogramInput, hist);

    return chart;
}

af_err af_draw_hist(const af_window window, const af_array X,
                    const double minval, const double maxval,
                    const af_cell* const props) {
    try {
        if (window == 0) { AF_ERROR("Not a valid window", AF_ERR_INTERNAL); }

        const ArrayInfo& Xinfo = getInfo(X);
        af_dtype Xtype         = Xinfo.getType();

        ARG_ASSERT(0, Xinfo.isVector());

        makeContextCurrent(window);

        fg_chart chart = NULL;

        switch (Xtype) {
            case f32:
                chart =
                    setup_histogram<float>(window, X, minval, maxval, props);
                break;
            case s32:
                chart = setup_histogram<int>(window, X, minval, maxval, props);
                break;
            case u32:
                chart = setup_histogram<uint>(window, X, minval, maxval, props);
                break;
            case s16:
                chart =
                    setup_histogram<short>(window, X, minval, maxval, props);
                break;
            case u16:
                chart =
                    setup_histogram<ushort>(window, X, minval, maxval, props);
                break;
            case u8:
                chart =
                    setup_histogram<uchar>(window, X, minval, maxval, props);
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
