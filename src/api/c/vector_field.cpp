/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/graphics.h>
#include <af/data.h>

#include <ArrayInfo.hpp>
#include <graphics_common.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <vector_field.hpp>
#include <reduce.hpp>
#include <transpose.hpp>
#include <handle.hpp>

using af::dim4;
using namespace detail;

#if defined(WITH_GRAPHICS)
using namespace graphics;

template<typename T>
forge::Chart* setup_vector_field(const forge::Window* const window,
                                 const af_array points, const af_array directions,
                                 const af_cell* const props, const bool transpose_ = true)
{
    Array<T> pIn = getArray<T>(points);
    Array<T> dIn = getArray<T>(directions);

    // do transpose if required
    if(transpose_) {
        pIn = transpose<T>(pIn, false);
        dIn = transpose<T>(dIn, false);
    }

    ForgeManager& fgMngr = ForgeManager::getInstance();

    // Get the chart for the current grid position (if any)
    forge::Chart* chart = NULL;

    if(pIn.dims()[0] == 2) {
        if (props->col>-1 && props->row>-1)
            chart = fgMngr.getChart(window, props->row, props->col, FG_CHART_2D);
        else
            chart = fgMngr.getChart(window, 0, 0, FG_CHART_2D);
    } else {
        if (props->col>-1 && props->row>-1)
            chart = fgMngr.getChart(window, props->row, props->col, FG_CHART_3D);
        else
            chart = fgMngr.getChart(window, 0, 0, FG_CHART_3D);
    }

    forge::VectorField* vectorfield = fgMngr.getVectorField(chart, pIn.dims()[1], getGLType<T>());

    vectorfield->setColor(1.0, 1.0, 0.0, 1.0);

    copy_vector_field<T>(pIn, dIn, vectorfield);

    return chart;
}

af_err vectorFieldWrapper(const af_window wind, const af_array points, const af_array directions,
                          const af_cell* const props)
{
    if(wind==0) {
        AF_RETURN_ERROR("Not a valid window", AF_SUCCESS);
    }

    try {
        ArrayInfo pInfo = getInfo(points);
        af::dim4 pDims  = pInfo.dims();
        af_dtype pType  = pInfo.getType();

        ArrayInfo dInfo = getInfo(directions);
        af::dim4 dDims  = dInfo.dims();
        af_dtype dType  = dInfo.getType();

        DIM_ASSERT(0, pDims == dDims);
        DIM_ASSERT(0, pDims.ndims() == 2);
        DIM_ASSERT(0, pDims[1] == 2 || pDims[1] == 3); // Columns:P 2 means 2D and 3 means 3D

        TYPE_ASSERT(pType == dType);

        forge::Window* window = reinterpret_cast<forge::Window*>(wind);
        makeContextCurrent(window);

        forge::Chart* chart = NULL;

        switch(pType) {
            case f32: chart = setup_vector_field<float  >(window, points, directions, props); break;
            case s32: chart = setup_vector_field<int    >(window, points, directions, props); break;
            case u32: chart = setup_vector_field<uint   >(window, points, directions, props); break;
            case s16: chart = setup_vector_field<short  >(window, points, directions, props); break;
            case u16: chart = setup_vector_field<ushort >(window, points, directions, props); break;
            case u8 : chart = setup_vector_field<uchar  >(window, points, directions, props); break;
            default:  TYPE_ERROR(1, pType);
        }

        // Window's draw function requires either image or chart
        if (props->col > -1 && props->row > -1)
            window->draw(props->row, props->col, *chart, props->title);
        else
            window->draw(*chart);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err vectorFieldWrapper(const af_window wind,
                          const af_array xPoints, const af_array yPoints, const af_array zPoints,
                          const af_array xDirs, const af_array yDirs, const af_array zDirs,
                          const af_cell* const props)
{
    if(wind==0) {
        AF_RETURN_ERROR("Not a valid window", AF_SUCCESS);
    }

    try {
        ArrayInfo xpInfo = getInfo(xPoints);
        ArrayInfo ypInfo = getInfo(yPoints);
        ArrayInfo zpInfo = getInfo(zPoints);

        af::dim4 xpDims  = xpInfo.dims();
        af::dim4 ypDims  = ypInfo.dims();
        af::dim4 zpDims  = zpInfo.dims();

        af_dtype xpType  = xpInfo.getType();
        af_dtype ypType  = ypInfo.getType();
        af_dtype zpType  = zpInfo.getType();

        ArrayInfo xdInfo = getInfo(xDirs);
        ArrayInfo ydInfo = getInfo(yDirs);
        ArrayInfo zdInfo = getInfo(zDirs);

        af::dim4 xdDims  = xdInfo.dims();
        af::dim4 ydDims  = ydInfo.dims();
        af::dim4 zdDims  = zdInfo.dims();

        af_dtype xdType  = xdInfo.getType();
        af_dtype ydType  = ydInfo.getType();
        af_dtype zdType  = zdInfo.getType();

        // Assert all arrays are equal dimensions
        DIM_ASSERT(1, xpDims == xdDims);
        DIM_ASSERT(2, ypDims == ydDims);
        DIM_ASSERT(3, zpDims == zdDims);

        DIM_ASSERT(1, xpDims == ypDims);
        DIM_ASSERT(1, xpDims == zpDims);

        // Verify vector
        DIM_ASSERT(1, xpDims.ndims() == 1);

        // Assert all arrays are equal types
        DIM_ASSERT(1, xpType == xdType);
        DIM_ASSERT(2, ypType == ydType);
        DIM_ASSERT(3, zpType == zdType);

        DIM_ASSERT(1, xpType == ypType);
        DIM_ASSERT(1, xpType == zpType);

        forge::Window* window = reinterpret_cast<forge::Window*>(wind);
        makeContextCurrent(window);

        forge::Chart* chart = NULL;

        // Join for set up vector
        af_array points = 0, directions = 0;
        af_array pIn[] = {xPoints, yPoints, zPoints};
        af_array dIn[] = {xDirs, yDirs, zDirs};
        AF_CHECK(af_join_many(&points, 1, 3, pIn));
        AF_CHECK(af_join_many(&directions, 1, 3, dIn));

        switch(xpType) {
            case f32: chart = setup_vector_field<float  >(window, points, directions, props); break;
            case s32: chart = setup_vector_field<int    >(window, points, directions, props); break;
            case u32: chart = setup_vector_field<uint   >(window, points, directions, props); break;
            case s16: chart = setup_vector_field<short  >(window, points, directions, props); break;
            case u16: chart = setup_vector_field<ushort >(window, points, directions, props); break;
            case u8 : chart = setup_vector_field<uchar  >(window, points, directions, props); break;
            default:  TYPE_ERROR(1, xpType);
        }

        // Window's draw function requires either image or chart
        if (props->col > -1 && props->row > -1)
            window->draw(props->row, props->col, *chart, props->title);
        else
            window->draw(*chart);

        AF_CHECK(af_release_array(points));
        AF_CHECK(af_release_array(directions));
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err vectorFieldWrapper(const af_window wind,
                          const af_array xPoints, const af_array yPoints,
                          const af_array xDirs, const af_array yDirs,
                          const af_cell* const props)
{
    if(wind==0) {
        AF_RETURN_ERROR("Not a valid window", AF_SUCCESS);
    }

    try {
        ArrayInfo xpInfo = getInfo(xPoints);
        ArrayInfo ypInfo = getInfo(yPoints);

        af::dim4 xpDims  = xpInfo.dims();
        af::dim4 ypDims  = ypInfo.dims();

        af_dtype xpType  = xpInfo.getType();
        af_dtype ypType  = ypInfo.getType();

        ArrayInfo xdInfo = getInfo(xDirs);
        ArrayInfo ydInfo = getInfo(yDirs);

        af::dim4 xdDims  = xdInfo.dims();
        af::dim4 ydDims  = ydInfo.dims();

        af_dtype xdType  = xdInfo.getType();
        af_dtype ydType  = ydInfo.getType();

        // Assert all arrays are equal dimensions
        DIM_ASSERT(1, xpDims == xdDims);
        DIM_ASSERT(2, ypDims == ydDims);

        DIM_ASSERT(1, xpDims == ypDims);

        // Verify vector
        DIM_ASSERT(1, xpDims.ndims() == 1);

        // Assert all arrays are equal types
        DIM_ASSERT(1, xpType == xdType);
        DIM_ASSERT(2, ypType == ydType);

        DIM_ASSERT(1, xpType == ypType);

        forge::Window* window = reinterpret_cast<forge::Window*>(wind);
        makeContextCurrent(window);

        forge::Chart* chart = NULL;

        // Join for set up vector
        af_array points = 0, directions = 0;
        AF_CHECK(af_join(&points, 1, xPoints, yPoints));
        AF_CHECK(af_join(&directions, 1, xDirs, yDirs));

        switch(xpType) {
            case f32: chart = setup_vector_field<float  >(window, points, directions, props); break;
            case s32: chart = setup_vector_field<int    >(window, points, directions, props); break;
            case u32: chart = setup_vector_field<uint   >(window, points, directions, props); break;
            case s16: chart = setup_vector_field<short  >(window, points, directions, props); break;
            case u16: chart = setup_vector_field<ushort >(window, points, directions, props); break;
            case u8 : chart = setup_vector_field<uchar  >(window, points, directions, props); break;
            default:  TYPE_ERROR(1, xpType);
        }

        // Window's draw function requires either image or chart
        if (props->col > -1 && props->row > -1)
            window->draw(props->row, props->col, *chart, props->title);
        else
            window->draw(*chart);

        AF_CHECK(af_release_array(points));
        AF_CHECK(af_release_array(directions));
    }
    CATCHALL;
    return AF_SUCCESS;
}

#endif // WITH_GRAPHICS

// ADD THIS TO UNIFIED
af_err af_draw_vector_field_nd(const af_window wind,
                const af_array points, const af_array directions,
                const af_cell* const props)
{
#if defined(WITH_GRAPHICS)
    return vectorFieldWrapper(wind, points, directions, props);
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}

af_err af_draw_vector_field_3d(
                const af_window wind,
                const af_array xPoints, const af_array yPoints, const af_array zPoints,
                const af_array xDirs, const af_array yDirs, const af_array zDirs,
                const af_cell* const props)
{
#if defined(WITH_GRAPHICS)
    return vectorFieldWrapper(wind, xPoints, yPoints, zPoints, xDirs, yDirs, zDirs, props);
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}

af_err af_draw_vector_field_2d(
                const af_window wind,
                const af_array xPoints, const af_array yPoints,
                const af_array xDirs, const af_array yDirs,
                const af_cell* const props)
{
#if defined(WITH_GRAPHICS)
    return vectorFieldWrapper(wind, xPoints, yPoints, xDirs, yDirs, props);
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}
