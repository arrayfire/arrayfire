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

#include <common/ArrayInfo.hpp>
#include <common/graphics_common.hpp>
#include <common/err_common.hpp>
#include <backend.hpp>
#include <surface.hpp>
#include <reduce.hpp>
#include <join.hpp>
#include <tile.hpp>
#include <reorder.hpp>
#include <handle.hpp>

using af::dim4;
using namespace detail;

#if defined(WITH_GRAPHICS)
using namespace graphics;

template<typename T>
forge::Chart* setup_surface(const forge::Window* const window,
                            const af_array xVals, const af_array yVals, const af_array zVals,
                            const af_cell* const props)
{
    Array<T> xIn = getArray<T>(xVals);
    Array<T> yIn = getArray<T>(yVals);
    Array<T> zIn = getArray<T>(zVals);

    const ArrayInfo& Xinfo = getInfo(xVals);
    const ArrayInfo& Yinfo = getInfo(yVals);
    const ArrayInfo& Zinfo = getInfo(zVals);

    af::dim4 X_dims = Xinfo.dims();
    af::dim4 Y_dims = Yinfo.dims();
    af::dim4 Z_dims = Zinfo.dims();

    if(Xinfo.isVector()){
        // Convert xIn is a column vector
        xIn = modDims(xIn, xIn.elements());
        // Now tile along second dimension
        dim4 x_tdims(1, Y_dims[0], 1, 1);
        xIn = tile(xIn, x_tdims);

        // Convert yIn to a row vector
        yIn= modDims(yIn, af::dim4(1, yIn.elements()));
        // Now tile along first dimension
        dim4 y_tdims(X_dims[0], 1, 1, 1);
        yIn = tile(yIn, y_tdims);
    }

    // Flatten xIn, yIn and zIn into row vectors
    dim4 rowDims = dim4(1, zIn.elements());
    xIn = modDims(xIn, rowDims);
    yIn = modDims(yIn, rowDims);
    zIn = modDims(zIn, rowDims);

    // Now join along first dimension, skip reorder
    std::vector<Array<T> > inputs{xIn, yIn, zIn};
    Array<T> Z = join(0, inputs);

    ForgeManager& fgMngr = ForgeManager::getInstance();

    // Get the chart for the current grid position (if any)
    forge::Chart* chart = NULL;
    if (props->col>-1 && props->row>-1)
        chart = fgMngr.getChart(window, props->row, props->col, FG_CHART_3D);
    else
        chart = fgMngr.getChart(window, 0, 0, FG_CHART_3D);

    forge::Surface* surface = fgMngr.getSurface(chart, Z_dims[0], Z_dims[1], getGLType<T>());

    surface->setColor(0.0, 1.0, 0.0, 1.0);

    // If chart axes limits do not have a manual override
    // then compute and set axes limits
    if(!fgMngr.getChartAxesOverride(chart)) {
        float cmin[3], cmax[3];
        T     dmin[3], dmax[3];
        chart->getAxesLimits(&cmin[0], &cmax[0], &cmin[1], &cmax[1], &cmin[2], &cmax[2]);
        dmin[0] = reduce_all<af_min_t, T, T>(xIn);
        dmax[0] = reduce_all<af_max_t, T, T>(xIn);
        dmin[1] = reduce_all<af_min_t, T, T>(yIn);
        dmax[1] = reduce_all<af_max_t, T, T>(yIn);
        dmin[2] = reduce_all<af_min_t, T, T>(zIn);
        dmax[2] = reduce_all<af_max_t, T, T>(zIn);

        if(cmin[0] == 0 && cmax[0] == 0
        && cmin[1] == 0 && cmax[1] == 0
        && cmin[2] == 0 && cmax[2] == 0) {
            // No previous limits. Set without checking
            cmin[0] = step_round(dmin[0], false);
            cmax[0] = step_round(dmax[0], true);
            cmin[1] = step_round(dmin[1], false);
            cmax[1] = step_round(dmax[1], true);
            cmin[2] = step_round(dmin[2], false);
            cmax[2] = step_round(dmax[2], true);
        } else {
            if(cmin[0] > dmin[0]) cmin[0] = step_round(dmin[0], false);
            if(cmax[0] < dmax[0]) cmax[0] = step_round(dmax[0], true);
            if(cmin[1] > dmin[1]) cmin[1] = step_round(dmin[1], false);
            if(cmax[1] < dmax[1]) cmax[1] = step_round(dmax[1], true);
            if(cmin[2] > dmin[2]) cmin[2] = step_round(dmin[2], false);
            if(cmax[2] < dmax[2]) cmax[2] = step_round(dmax[2], true);
        }

        chart->setAxesLimits(cmin[0], cmax[0], cmin[1], cmax[1], cmin[2], cmax[2]);
    }

    copy_surface<T>(Z, surface);

    return chart;
}
#endif

af_err af_draw_surface(const af_window wind, const af_array xVals, const af_array yVals, const af_array S, const af_cell* const props)
{
#if defined(WITH_GRAPHICS)
    if(wind==0) {
        std::cerr<<"Not a valid window"<<std::endl;
        return AF_SUCCESS;
    }

    try {
        const ArrayInfo& Xinfo = getInfo(xVals);
        af::dim4 X_dims = Xinfo.dims();
        af_dtype Xtype  = Xinfo.getType();

        const ArrayInfo& Yinfo = getInfo(yVals);
        af::dim4 Y_dims = Yinfo.dims();
        af_dtype Ytype  = Yinfo.getType();

        const ArrayInfo& Sinfo = getInfo(S);
        af::dim4 S_dims = Sinfo.dims();
        af_dtype Stype  = Sinfo.getType();

        TYPE_ASSERT(Xtype == Ytype);
        TYPE_ASSERT(Ytype == Stype);

        if(!Yinfo.isVector()){
            DIM_ASSERT(1, X_dims == Y_dims);
            DIM_ASSERT(3, Y_dims == S_dims);
        }else{
            DIM_ASSERT(3, ( X_dims[0] * Y_dims[0] == (dim_t)Sinfo.elements()));
        }

        forge::Window* window = static_cast<forge::Window*>(wind);
        makeContextCurrent(window);

        forge::Chart* chart = NULL;

        switch(Xtype) {
            case f32: chart = setup_surface<float  >(window, xVals, yVals , S, props); break;
            case s32: chart = setup_surface<int    >(window, xVals, yVals , S, props); break;
            case u32: chart = setup_surface<uint   >(window, xVals, yVals , S, props); break;
            case s16: chart = setup_surface<short  >(window, xVals, yVals , S, props); break;
            case u16: chart = setup_surface<ushort >(window, xVals, yVals , S, props); break;
            case u8 : chart = setup_surface<uchar  >(window, xVals, yVals , S, props); break;
            default:  TYPE_ERROR(1, Xtype);
        }

        auto gridDims = ForgeManager::getInstance().getWindowGrid(window);
        if (props->col>-1 && props->row>-1)
            window->draw(gridDims.first, gridDims.second,
                         props->row * gridDims.second + props->col,
                         *chart, props->title);
        else
            window->draw(*chart);
    }
    CATCHALL;
    return AF_SUCCESS;
#else
    UNUSED(wind);
    UNUSED(xVals);
    UNUSED(yVals);
    UNUSED(S);
    UNUSED(props);
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}
