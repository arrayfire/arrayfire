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

#include <ArrayInfo.hpp>
#include <graphics_common.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <plot.hpp>
#include <reduce.hpp>
#include <join.hpp>
#include <reorder.hpp>
#include <handle.hpp>

using af::dim4;
using namespace detail;

#if defined(WITH_GRAPHICS)
using namespace graphics;

template<typename T>
fg::Plot* setup_plot(const af_array X, const af_array Y)
{
    Array<T> xIn = getArray<T>(X);
    Array<T> yIn = getArray<T>(Y);

    T xmax = reduce_all<af_max_t, T, T>(xIn);
    T xmin = reduce_all<af_min_t, T, T>(xIn);
    T ymax = reduce_all<af_max_t, T, T>(yIn);
    T ymin = reduce_all<af_min_t, T, T>(yIn);

    dim4 rdims(1, 0, 2, 3);

    Array<T> Z = join(1, xIn, yIn);
    Array<T> P = reorder(Z, rdims);

    ArrayInfo Xinfo = getInfo(X);
    af::dim4 X_dims = Xinfo.dims();

    ForgeManager& fgMngr = ForgeManager::getInstance();
    fg::Plot* plot = fgMngr.getPlot(X_dims.elements(), getGLType<T>());
    plot->setColor(1.0, 0.0, 0.0);
    plot->setAxesLimits(xmax, xmin, ymax, ymin);
    plot->setXAxisTitle("X Axis");
    plot->setYAxisTitle("Y Axis");

    copy_plot<T>(P, plot);

    return plot;
}
#endif

af_err af_draw_plot(const af_window wind, const af_array X, const af_array Y, const af_cell* const props)
{
#if defined(WITH_GRAPHICS)
    if(wind==0) {
        std::cerr<<"Not a valid window"<<std::endl;
        return AF_SUCCESS;
    }

    try {
        ArrayInfo Xinfo = getInfo(X);
        af::dim4 X_dims = Xinfo.dims();
        af_dtype Xtype  = Xinfo.getType();

        ArrayInfo Yinfo = getInfo(Y);
        af::dim4 Y_dims = Yinfo.dims();
        af_dtype Ytype  = Yinfo.getType();

        DIM_ASSERT(0, X_dims == Y_dims);
        DIM_ASSERT(0, X_dims == Y_dims);
        DIM_ASSERT(0, Xinfo.isVector());

        TYPE_ASSERT(Xtype == Ytype);

        fg::Window* window = reinterpret_cast<fg::Window*>(wind);
        window->makeCurrent();
        fg::Plot* plot = NULL;

        switch(Xtype) {
            case f32: plot = setup_plot<float  >(X, Y); break;
            case s32: plot = setup_plot<int    >(X, Y); break;
            case u32: plot = setup_plot<uint   >(X, Y); break;
            case u8 : plot = setup_plot<uchar  >(X, Y); break;
            default:  TYPE_ERROR(1, Xtype);
        }

        if (props->col>-1 && props->row>-1)
            window->draw(props->col, props->row, *plot, props->title);
        else
            window->draw(*plot);
    }
    CATCHALL;
    return AF_SUCCESS;
#else
    return AF_ERR_NO_GFX;
#endif
}
