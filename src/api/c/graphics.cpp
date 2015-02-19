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
#include <graphics.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline void draw_image(const af_array in, const ImageHandle &image)
{
    draw_image(getArray<T>(in), image);
}

static af_array convert_data(const af_array in, const ArrayInfo &info, const ImageHandle &image)
{
    af_array X = 0;
    af_array Y = 0;
    af_array Z = 0;
    // Tile if needed
    // Interleave values and transpose

    dim_type i2 = info.dims()[2];
    dim_type o2 = image->window->mode;

    if (i2 == 1) {
        if (o2 == 1) {
            af_reorder(&X, in, 2, 1, 0, 3);
        } else if (o2 == 3) {
            af_tile(&Y, in, 1, 1, 3, 1);
            af_reorder(&X, Y, 2, 1, 0, 3);
        } else if (o2 == 4) {
            af_array c1;
            af_constant(&c1, 1, 2, info.dims().get(), info.getType());
            af_tile(&Y, in, 1, 1, 3, 1);
            af_join(&Z, 2, Y, c1);
            af_reorder(&X, Z, 2, 1, 0, 3);
            if(c1!= 0) af_destroy_array(c1);
        }
    } else if (i2 == 3) {
        if (o2 == 1) {
            af_rgb2gray(&Y, in, 0.2126f, 0.7152f, 0.0722f);
            af_reorder(&X, Y, 2, 1, 0, 3);
        } else if (o2 == 3) {
            af_reorder(&X, in, 2, 1, 0, 3);
        } else if (o2 == 4) {
            af_array c1;
            af_constant(&c1, 1, 2, info.dims().get(), info.getType());
            af_join(&Y, 2, in, c1);
            af_reorder(&X, Y, 2, 1, 0, 3);
            if(c1!= 0) af_destroy_array(c1);
        }
    } else if (i2 == 4) {
        af_seq s[3] = {af_span, af_span, {0, 2, 1}};
        if (o2 == 1) {
            af_index(&Y, in, 3, s);
            af_rgb2gray(&Z, Y, 0.2126f, 0.7152f, 0.0722f);
            af_reorder(&X, Z, 2, 1, 0, 3);
            //FIXME Use Colorspace Conversion
        } else if (o2 == 3) {
            af_index(&Y, in, 3, s);
            af_reorder(&X, Y, 2, 1, 0, 3);
        } else if (o2 == 4) {
            af_reorder(&X, in, 2, 1, 0, 3);
        }
    }

    if(Y != 0) af_destroy_array(Y);
    if(Z != 0) af_destroy_array(Z);

    return X;
}

af_err af_draw_image(const af_array in, const ImageHandle &image)
{
    try {
        ArrayInfo info = getInfo(in);

        af::dim4 in_dims = info.dims();
        af_dtype type    = info.getType();
        DIM_ASSERT(0, in_dims[2] == 1 || in_dims[2] == 3);   // Correct Number of Channels
        DIM_ASSERT(0, in_dims[3] == 1);

        // Test to make sure window GLenum type and in.type() are compatible
        ARG_ASSERT(0, type == f32);

        af_array X = convert_data(in, info, image);

        afgfx_make_window_current(image->window);

        switch(type) {
            case f32: draw_image<float  >(X, image);  break;
            case f64: draw_image<double >(X, image);  break;
            case b8:  draw_image<char   >(X, image);  break;
            case s32: draw_image<int    >(X, image);  break;
            case u32: draw_image<uint   >(X, image);  break;
            case u8:  draw_image<uchar  >(X, image);  break;
            default:  TYPE_ERROR(1, type);
        }

        afgfx_draw_image(image);

        if(X != 0) af_destroy_array(X);
    }
    CATCHALL;

    return AF_SUCCESS;
}

#endif
