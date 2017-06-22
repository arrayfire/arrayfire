/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/graphics.h>
#include <common/graphics_common.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <backend.hpp>
#include <reduce.hpp>
#include <cast.hpp>
#include <handle.hpp>
#include <hist_graphics.hpp>

using af::dim4;
using namespace detail;

#if defined(WITH_GRAPHICS)
using namespace graphics;

template<typename T>
forge::Chart* setup_histogram(const forge::Window* const window,
                              const af_array in, const double minval, const double maxval,
                              const af_cell* const props)
{
    Array<T> histogramInput = getArray<T>(in);
    dim_t nBins = histogramInput.elements();

    // Retrieve Forge Histogram with nBins and array type
    ForgeManager& fgMngr = ForgeManager::getInstance();

    // Get the chart for the current grid position (if any)
    forge::Chart* chart = NULL;
    if (props->col>-1 && props->row>-1)
        chart = fgMngr.getChart(window, props->row, props->col, FG_CHART_2D);
    else
        chart = fgMngr.getChart(window, 0, 0, FG_CHART_2D);

    // Create a histogram for the chart
    forge::Histogram* hist = fgMngr.getHistogram(chart, nBins, getGLType<T>());

    // Set histogram bar colors to orange
    hist->setColor(0.929f, 0.486f, 0.2745f, 1.0f);

    // If chart axes limits do not have a manual override
    // then compute and set axes limits
    if(!fgMngr.getChartAxesOverride(chart)) {
        float xMin, xMax, yMin, yMax;
        chart->getAxesLimits(&xMin, &xMax, &yMin, &yMax);
        T freqMax = detail::reduce_all<af_max_t, T, T>(histogramInput);

        if(xMin == 0 && xMax == 0 && yMin == 0 && yMax == 0) {
            // No previous limits. Set without checking
            xMin = step_round(minval, false);
            xMax = step_round(maxval, true);
            yMax = step_round(freqMax, true);
            // For histogram, always set yMin to 0.
            yMin = 0;
        } else {
            if(xMin > minval)  xMin = step_round(minval, false);
            if(xMax < maxval)  xMax = step_round(maxval, true);
            if(yMax < freqMax) yMax = step_round(freqMax, true);
            // For histogram, always set yMin to 0.
            yMin = 0;
        }

        chart->setAxesLimits(xMin, xMax, yMin, yMax);
    }

    copy_histogram<T>(histogramInput, hist);

    return chart;
}
#endif

af_err af_draw_hist(const af_window wind, const af_array X, const double minval, const double maxval,
                    const af_cell* const props)
{
#if defined(WITH_GRAPHICS)
    if(wind==0) {
        std::cerr<<"Not a valid window"<<std::endl;
        return AF_SUCCESS;
    }

    try {
        const ArrayInfo& Xinfo = getInfo(X);
        af_dtype Xtype  = Xinfo.getType();

        ARG_ASSERT(0, Xinfo.isVector());

        forge::Window* window = reinterpret_cast<forge::Window*>(wind);
        makeContextCurrent(window);

        forge::Chart* chart = NULL;

        switch(Xtype) {
            case f32: chart = setup_histogram<float  >(window, X, minval, maxval, props); break;
            case s32: chart = setup_histogram<int    >(window, X, minval, maxval, props); break;
            case u32: chart = setup_histogram<uint   >(window, X, minval, maxval, props); break;
            case s16: chart = setup_histogram<short  >(window, X, minval, maxval, props); break;
            case u16: chart = setup_histogram<ushort >(window, X, minval, maxval, props); break;
            case u8 : chart = setup_histogram<uchar  >(window, X, minval, maxval, props); break;
            default:  TYPE_ERROR(1, Xtype);
        }

        auto gridDims = ForgeManager::getInstance().getWindowGrid(window);
        // Window's draw function requires either image or chart
        if (props->col > -1 && props->row > -1)
            window->draw(gridDims.first, gridDims.second, props->col * gridDims.first + props->row,
                         *chart, props->title);
        else
            window->draw(*chart);
    }
    CATCHALL;
    return AF_SUCCESS;
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}
