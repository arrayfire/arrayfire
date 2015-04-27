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
#include <af/index.h>
#include <af/data.h>

#include <ArrayInfo.hpp>
#include <graphics_common.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <image.hpp>
#include <handle.hpp>
#include <reorder.hpp>
#include <tile.hpp>
#include <join.hpp>

#include <iostream>

using af::dim4;
using namespace detail;
using namespace graphics;

#if defined(WITH_GRAPHICS)
template<typename T>
static fg::Image* convert_and_copy_image(const af_array in)
{
    ArrayInfo info      = getInfo(in);
    const Array<T> _in  = getArray<T>(in);
    dim4 inDims = _in.dims();

    dim4 rdims = (inDims[2]>1 ? dim4(2, 1, 0, 3) : dim4(1, 0, 2, 3));

    Array<T> imgData = reorder(_in, rdims);

    ForgeManager& fgMngr = ForgeManager::getInstance();

    fg::Image* ret_val = fgMngr.getImage(inDims[1], inDims[0], (fg::ColorMode)inDims[2], getGLType<T>());

    copy_image<T>(imgData, ret_val);

    return ret_val;
}
#endif

af_err af_draw_image(const af_array in)
{
#if defined(WITH_GRAPHICS)
    try {
        ArrayInfo info = getInfo(in);

        af::dim4 in_dims = info.dims();
        af_dtype type    = info.getType();
        DIM_ASSERT(0, in_dims[2] == 1 || in_dims[2] == 3 || in_dims[2] == 4);
        DIM_ASSERT(0, in_dims[3] == 1);

        fg::Window* window = ForgeManager::getWindow();
        fg::makeCurrent(window);
        fg::Image* image = NULL;

        switch(type) {
            case f32: image = convert_and_copy_image<float>(in); break;
            case b8 : image = convert_and_copy_image<char >(in); break;
            case s32: image = convert_and_copy_image<int  >(in); break;
            case u32: image = convert_and_copy_image<uint >(in); break;
            case u8 : image = convert_and_copy_image<uchar>(in); break;
            default:  TYPE_ERROR(1, type);
        }

        window->draw(*image);
    }
    CATCHALL;

    return AF_SUCCESS;
#else
    return AF_ERR_NO_GRAPHICS;
#endif
}
