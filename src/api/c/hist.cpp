/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/graphics.h>
#include <graphics_common.hpp>
#include <ArrayInfo.hpp>
#include <err_common.hpp>
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

    T freqMax = detail::reduce_all<af_max_t, T, T>(histogramInput);

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

    // set x axis limits to maximum and minimum values of data
    // and y axis limits to range [0, nBins]
    chart->setAxesLimits(minval, maxval, 0.0f, double(freqMax));
    chart->setAxesTitles("Bins", "Frequency");

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
        ArrayInfo Xinfo = getInfo(X);
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

        // Window's draw function requires either image or chart
        if (props->col > -1 && props->row > -1)
            window->draw(props->col, props->row, *chart, props->title);
        else
            window->draw(*chart);
    }
    CATCHALL;
    return AF_SUCCESS;
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}
