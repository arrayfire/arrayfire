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
fg::Histogram* setup_histogram(const af_array in, const double minval, const double maxval)
{
    Array<T> histogramInput = getArray<T>(in);
    dim_t nBins = histogramInput.elements();

    T freqMax = detail::reduce_all<af_max_t, T, T>(histogramInput);

    /* retrieve Forge Histogram with nBins and array type */
    ForgeManager& fgMngr = ForgeManager::getInstance();
    fg::Histogram* hist = fgMngr.getHistogram(nBins, getGLType<T>());
    /* set histogram bar colors to orange */
    hist->setBarColor(0.929f, 0.486f, 0.2745f);
    /* set x axis limits to maximum and minimum values of data
     * and y axis limits to range [0, nBins]*/
    hist->setAxesLimits(maxval, minval, double(freqMax), 0.0f);
    hist->setXAxisTitle("Bins");
    hist->setYAxisTitle("Frequency");

    copy_histogram<T>(histogramInput, hist);

    return hist;
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

        fg::Window* window = reinterpret_cast<fg::Window*>(wind);
        window->makeCurrent();
        fg::Histogram* hist = NULL;

        switch(Xtype) {
            case f32: hist = setup_histogram<float  >(X, minval, maxval); break;
            case s32: hist = setup_histogram<int    >(X, minval, maxval); break;
            case u32: hist = setup_histogram<uint   >(X, minval, maxval); break;
            case u8 : hist = setup_histogram<uchar  >(X, minval, maxval); break;
            default:  TYPE_ERROR(1, Xtype);
        }

        if (props->col>-1 && props->row>-1)
            window->draw(props->col, props->row, *hist, props->title);
        else
            window->draw(*hist);
    }
    CATCHALL;
    return AF_SUCCESS;
#else
    return AF_ERR_NO_GFX;
#endif
}
