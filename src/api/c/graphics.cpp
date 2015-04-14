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
#include <af/index.h>
#include <af/data.h>

#include <ArrayInfo.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <graphics.hpp>
#include <handle.hpp>
#include <reorder.hpp>
#include <tile.hpp>
#include <join.hpp>

using af::dim4;
using namespace detail;

template<typename T, int i2, int o2>
static void convert_and_copy_image(const af_array in, const fg_image_handle image)
{
    ArrayInfo info = getInfo(in);

    const Array<T> _in = getArray<T>(in);

    Array<T> X = createEmptyArray<T>(dim4());
    Array<T> Y = createEmptyArray<T>(dim4());
    Array<T> Z = createEmptyArray<T>(dim4());

    dim4 rdims(2, 1, 0, 3);
    dim4 tdims(1, 1, 3, 1);

    if (i2 == 1) {
        if (o2 == 1) {
            X = reorder(_in, rdims);
        } else if (o2 == 3) {
            Y = tile(_in, tdims);
            X = reorder(Y, rdims);
        } else if (o2 == 4) {
            Array<T> c1 = createValueArray<T>(info.dims(), 1);
            Z = tile(_in, tdims);
            Y = join(2, Z, c1);
            X = reorder(Y, rdims);
        }
    } else if (i2 == 3) {
        if (o2 == 1) {
            af_array y = 0;
            AF_CHECK(af_rgb2gray(&y, in, 0.2126f, 0.7152f, 0.0722f));
            Y = getArray<T>(y);
            X = reorder(Y, rdims);
            if(y != 0) AF_CHECK(af_destroy_array(y));
        } else if (o2 == 3) {
            X = reorder(_in, rdims);
        } else if (o2 == 4) {
            Array<T> c1 = createValueArray<T>(info.dims(), 1);
            Y = join(2, _in, c1);
            X = reorder(Y, rdims);
        }
    } else if (i2 == 4) {

        af_seq s[3] = {af_span, af_span, {0, 2, 1}};
        std::vector<af_seq> s_vec(s, s + sizeof(s) / sizeof(s[0]));

        if (o2 == 1) {
            af_array y = 0;
            af_array z = 0;
            Z = createSubArray(_in, s_vec, false);
            z = getHandle<T>(Z);
            AF_CHECK(af_rgb2gray(&y, z, 0.2126f, 0.7152f, 0.0722f));
            Y = getArray<T>(y);
            X = reorder(Y, rdims);
            if(y != 0) AF_CHECK(af_destroy_array(y));
        } else if (o2 == 3) {
            Y = createSubArray(_in, s_vec, false);
            X = reorder(Y, rdims);
        } else if (o2 == 4) {
            X = reorder(_in, rdims);
        }
    }

    copy_image<T>(X, image);

}

template<typename T, int i2>
static void convert_and_copy_image(const af_array in, const fg_image_handle image)
{
    dim_type o2 = image->window->mode;
    switch(o2) {
        case 1: convert_and_copy_image<T, i2, 1>(in, image); break;
        case 3: convert_and_copy_image<T, i2, 3>(in, image); break;
        case 4: convert_and_copy_image<T, i2, 4>(in, image); break;
    }
}

template<typename T>
static void convert_and_copy_image(const af_array in, const fg_image_handle image)
{
    ArrayInfo info = getInfo(in);
    dim_type i2 = info.dims()[2];
    switch(i2) {
        case 1: convert_and_copy_image<T, 1>(in, image); break;
        case 3: convert_and_copy_image<T, 3>(in, image); break;
        case 4: convert_and_copy_image<T, 4>(in, image); break;
    }
}

af_err af_draw_image(const af_array in, const fg_image_handle image)
{
    try {
        ArrayInfo info = getInfo(in);

        af::dim4 in_dims = info.dims();
        af_dtype type    = info.getType();
        DIM_ASSERT(0, in_dims[2] == 1 || in_dims[2] == 3 || in_dims[2] == 4);
        DIM_ASSERT(0, in_dims[3] == 1);

        fg_make_window_current(image->window);

        switch(type) {
            case f32:
                convert_and_copy_image<float  >(in, image);
                break;
            case f64:
                convert_and_copy_image<double >(in, image);
                break;
            case b8:
                convert_and_copy_image<char   >(in, image);
                break;
            case s32:
                convert_and_copy_image<int    >(in, image);
                break;
            case u32:
                convert_and_copy_image<uint   >(in, image);
                break;
            case u8:
                convert_and_copy_image<uchar  >(in, image);
                break;
            default:  TYPE_ERROR(1, type);
        }

        fg_draw_image(image);
    }
    CATCHALL;

    return AF_SUCCESS;
}
template<typename T>
void setup_plot(const af_array X, const af_array Y, const fg_plot_handle plot)
{
    copy_plot<T>(getArray<T>(X), getArray<T>(Y), plot);
}

af_err af_draw_plot(const af_array X, const af_array Y, const fg_plot_handle plot)
{
    try {
#if 1
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
            case f32:
                setup_plot<float  >(X, Y, plot);
                break;
            case s32:
                setup_plot<int    >(X, Y, plot);
                break;
            case u32:
                setup_plot<uint   >(X, Y, plot);
                break;
            case u8:
                setup_plot<uchar  >(X, Y, plot);
                break;
            default:  TYPE_ERROR(1, Xtype);
        }
#else

#endif
    }
    CATCHALL;
    return AF_SUCCESS;
}

#endif
