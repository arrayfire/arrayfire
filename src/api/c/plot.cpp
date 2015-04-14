/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined (WITH_GRAPHICS)

#include <af/graphics.h>
#include <af/image.h>

#include <ArrayInfo.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <plot.hpp>
#include <handle.hpp>

using af::dim4;
using namespace detail;

template<typename T>
void setup_plot(const af_array X, const af_array Y, const fg_plot_handle plot)
{
    copy_plot<T>(getArray<T>(X), getArray<T>(Y), plot);
}

af_err af_draw_plot(const af_array X, const af_array Y, const fg_plot_handle plot)
{
    try {
        ArrayInfo Xinfo = getInfo(X);
        af::dim4 X_dims = Xinfo.dims();
        af_dtype Xtype    = Xinfo.getType();

        ArrayInfo Yinfo = getInfo(Y);
        af::dim4 Y_dims = Yinfo.dims();
        af_dtype Ytype    = Yinfo.getType();

        DIM_ASSERT(0, X_dims == Y_dims);
        DIM_ASSERT(0, X_dims == Y_dims);
        DIM_ASSERT(0, Xinfo.isVector());

        TYPE_ASSERT(Xtype == Ytype);

        switch(Xtype) {
            case f32: setup_plot<float  >(X, Y, plot); break;
            case s32: setup_plot<int    >(X, Y, plot); break;
            case u32: setup_plot<uint   >(X, Y, plot); break;
            case u8 : setup_plot<uchar  >(X, Y, plot); break;
            default:  TYPE_ERROR(1, Xtype);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

#endif

