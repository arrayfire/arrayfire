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

#include <common/ArrayInfo.hpp>
#include <common/graphics_common.hpp>
#include <common/err_common.hpp>
#include <backend.hpp>
#include <image.hpp>
#include <handle.hpp>
#include <reorder.hpp>
#include <tile.hpp>
#include <join.hpp>
#include <cast.hpp>
#include <arith.hpp>

#include <iostream>
#include <limits>

using af::dim4;
using namespace detail;

#if defined(WITH_GRAPHICS)
using namespace graphics;


template<typename T>
Array<T> normalizePerType(const Array<T>& in)
{
    Array<float> inFloat = cast<float, T>(in);

    Array<float> cnst = createValueArray<float>(in.dims(), 1.0 - 1.0e-6f);

    Array<float> scaled = arithOp<float, af_mul_t>(inFloat, cnst, in.dims());

    return cast<T, float>(scaled);
}

template<>
Array<float> normalizePerType<float>(const Array<float>& in)
{
    return in;
}

template<typename T>
static forge::Image* convert_and_copy_image(const af_array in)
{
    const Array<T> _in  = getArray<T>(in);
    dim4 inDims = _in.dims();

    dim4 rdims = (inDims[2]>1 ? dim4(2, 1, 0, 3) : dim4(1, 0, 2, 3));

    Array<T> imgData = reorder(_in, rdims);

    ForgeManager& fgMngr = ForgeManager::getInstance();

    // The inDims[2] * 100 is a hack to convert to forge::ChannelFormat
    // TODO Write a proper conversion function
    forge::Image* ret_val = fgMngr.getImage(inDims[1], inDims[0], (forge::ChannelFormat)(inDims[2] * 100), getGLType<T>());

    copy_image<T>(normalizePerType<T>(imgData), ret_val);

    return ret_val;
}
#endif

af_err af_draw_image(const af_window wind, const af_array in, const af_cell* const props)
{
#if defined(WITH_GRAPHICS)
    if(wind==0) {
        std::cerr<<"Not a valid window"<<std::endl;
        return AF_SUCCESS;
    }

    try {
        const ArrayInfo& info = getInfo(in);

        af::dim4 in_dims = info.dims();
        af_dtype type    = info.getType();
        DIM_ASSERT(0, in_dims[2] == 1 || in_dims[2] == 3 || in_dims[2] == 4);
        DIM_ASSERT(0, in_dims[3] == 1);

        forge::Window* window = reinterpret_cast<forge::Window*>(wind);
        makeContextCurrent(window);
        forge::Image* image = NULL;

        switch(type) {
            case f32: image = convert_and_copy_image<float >(in); break;
            case b8 : image = convert_and_copy_image<char  >(in); break;
            case s32: image = convert_and_copy_image<int   >(in); break;
            case u32: image = convert_and_copy_image<uint  >(in); break;
            case s16: image = convert_and_copy_image<short >(in); break;
            case u16: image = convert_and_copy_image<ushort>(in); break;
            case u8 : image = convert_and_copy_image<uchar >(in); break;
            default:  TYPE_ERROR(1, type);
        }

        auto gridDims = ForgeManager::getInstance().getWindowGrid(window);
        window->setColorMap((forge::ColorMap)props->cmap);
        if (props->col>-1 && props->row>-1)
            window->draw(gridDims.first, gridDims.second, props->col * gridDims.first + props->row,
                         *image, props->title);
        else
            window->draw(*image);
    }
    CATCHALL;

    return AF_SUCCESS;
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}
