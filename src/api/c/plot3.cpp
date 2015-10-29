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
#include <plot3.hpp>
#include <reduce.hpp>
#include <join.hpp>
#include <transpose.hpp>
#include <reorder.hpp>
#include <handle.hpp>
#include <af/data.h>
#include <iostream>

using af::dim4;
using namespace detail;

#if defined(WITH_GRAPHICS)
using namespace graphics;

template<typename T>
fg::Plot3* setup_plot3(const af_array P)
{
    Array<T> pIn = getArray<T>(P);
    ArrayInfo Pinfo = getInfo(P);
    af::dim4 P_dims = Pinfo.dims();

    DIM_ASSERT(0, Pinfo.ndims() == 1 || Pinfo.ndims() == 2);
    DIM_ASSERT(0, (P_dims[0] == 3 || P_dims[1] == 3) ||
                    (Pinfo.isVector() && P_dims[0]%3 == 0));

    if(Pinfo.isVector()){
        dim4 rdims(P_dims.elements()/3, 3, 1, 1);
        pIn.modDims(rdims);
        P_dims = pIn.dims();
    }

    T max[3], min[3];
    if(P_dims[0] == 3) {
        af_get_data_ptr(max, getHandle(reduce<af_max_t, T, T>(pIn, 1)));
        af_get_data_ptr(min, getHandle(reduce<af_min_t, T, T>(pIn, 1)));
    }

    if(P_dims[1] == 3) {
        af_get_data_ptr(max, getHandle(reduce<af_max_t, T, T>(pIn, 0)));
        af_get_data_ptr(min, getHandle(reduce<af_min_t, T, T>(pIn, 0)));
    }

    ForgeManager& fgMngr = ForgeManager::getInstance();
    fg::Plot3* plot3 = fgMngr.getPlot3(P_dims.elements()/3, getGLType<T>());
    plot3->setColor(1.0, 0.0, 0.0);
    plot3->setAxesLimits(max[0], min[0],
                         max[1], min[1],
                         max[2], min[2]);
    plot3->setAxesTitles("X Axis", "Y Axis", "Z Axis");

    if(P_dims[1] == 3){
        pIn = transpose(pIn, false);
    }
    copy_plot3<T>(pIn, plot3);

    return plot3;
}
#endif

af_err af_draw_plot3(const af_window wind, const af_array P, const af_cell* const props)
{
#if defined(WITH_GRAPHICS)
    if(wind==0) {
        std::cerr<<"Not a valid window"<<std::endl;
        return AF_SUCCESS;
    }

    try {
        ArrayInfo Pinfo = getInfo(P);
        af_dtype Ptype  = Pinfo.getType();

        fg::Window* window = reinterpret_cast<fg::Window*>(wind);
        window->makeCurrent();
        fg::Plot3* plot3 = NULL;

        switch(Ptype) {
            case f32: plot3 = setup_plot3<float>(P); break;
            case s32: plot3 = setup_plot3<int  >(P); break;
            case u32: plot3 = setup_plot3<uint >(P); break;
            case u8 : plot3 = setup_plot3<uchar>(P); break;
            default:  TYPE_ERROR(1, Ptype);
        }

        if (props->col>-1 && props->row>-1)
            window->draw(props->col, props->row, *plot3, props->title);
        else
            window->draw(*plot3);
    }
    CATCHALL;
    return AF_SUCCESS;
#else
    return AF_ERR_NO_GFX;
#endif
}
