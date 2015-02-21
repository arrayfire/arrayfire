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

template<typename T>
static inline void convert_and_copy_image(const af_array in, const afgfx_image image)
{
    ArrayInfo info = getInfo(in);

    const Array<T> _in = getArray<T>(in);

    Array<T> *X = createEmptyArray<T>(dim4());
    Array<T> *Y = createEmptyArray<T>(dim4());
    Array<T> *Z = createEmptyArray<T>(dim4());
    af_array x = 0;
    af_array y = 0;
    af_array z = 0;

    dim_type i2 = info.dims()[2];
    dim_type o2 = image->window->mode;
    dim4 rdims(2, 1, 0, 3);
    dim4 tdims(1, 1, 3, 1);

    if (i2 == 1) {
        if (o2 == 1) {
            X = reorder(_in, rdims);
        } else if (o2 == 3) {
            Y = tile(_in, tdims);
            X = reorder(*Y, rdims);
        } else if (o2 == 4) {
            Array<T> *c1 = createValueArray<T>(info.dims(), 1);
            Z = tile(_in, tdims);
            Y = join(2, *Z, *c1);
            X = reorder(*Y, rdims);
            destroyArray<T>(*c1);
        }
    } else if (i2 == 3) {
        if (o2 == 1) {
            AF_CHECK(af_rgb2gray(&y, in, 0.2126f, 0.7152f, 0.0722f));
            *Y = getArray<T>(y);
            X = reorder(*Y, rdims);
        } else if (o2 == 3) {
            X = reorder(_in, rdims);
        } else if (o2 == 4) {
            Array<T> *c1 = createValueArray<T>(info.dims(), 1);
            Y = join(2, _in, *c1);
            X = reorder(*Y, rdims);
            destroyArray<T>(*c1);
        }
    } else if (i2 == 4) {
        af_seq s[3] = {af_span, af_span, {0, 2, 1}};
        if (o2 == 1) {
            AF_CHECK(af_index(&z, in, 3, s));
            AF_CHECK(af_rgb2gray(&y, z, 0.2126f, 0.7152f, 0.0722f));
            *Y = getArray<T>(y);
            X = reorder(*Y, rdims);
        } else if (o2 == 3) {
            AF_CHECK(af_index(&y, in, 3, s));
            *Y = getArray<T>(y);
            X = reorder(*Y, rdims);
        } else if (o2 == 4) {
            X = reorder(_in, rdims);
        }
    }

    copy_image<T>(*X, image);

    destroyArray<T>(*X);
    destroyArray<T>(*Y);
    destroyArray<T>(*Z);
    if(x != 0) AF_CHECK(af_destroy_array(x));
    if(y != 0) AF_CHECK(af_destroy_array(y));
    if(z != 0) AF_CHECK(af_destroy_array(z));
}

af_err af_draw_image(const af_array in, const afgfx_image image)
{
    try {
        ArrayInfo info = getInfo(in);

        af::dim4 in_dims = info.dims();
        af_dtype type    = info.getType();
        DIM_ASSERT(0, in_dims[2] == 1 || in_dims[2] == 3);   // Correct Number of Channels
        DIM_ASSERT(0, in_dims[3] == 1);

        // Test to make sure window GLenum type and in.type() are compatible
        ARG_ASSERT(0, type == f32);

        afgfx_make_window_current(image->window);

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

        afgfx_draw_image(image);
    }
    CATCHALL;

    return AF_SUCCESS;
}

#endif
