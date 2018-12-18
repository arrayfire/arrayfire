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

#include <limits>

using af::dim4;
using namespace detail;
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
static fg_image convert_and_copy_image(const af_array in)
{
    const Array<T> _in  = getArray<T>(in);
    dim4 inDims = _in.dims();

    dim4 rdims = (inDims[2]>1 ? dim4(2, 1, 0, 3) : dim4(1, 0, 2, 3));

    Array<T> imgData = reorder(_in, rdims);

    ForgeManager& fgMngr = ForgeManager::getInstance();

    // The inDims[2] * 100 is a hack to convert to fg_channel_format
    // TODO Write a proper conversion function
    fg_image ret_val = fgMngr.getImage(inDims[1], inDims[0],
                                       (fg_channel_format)(inDims[2] * 100),
                                       getGLType<T>());
    copy_image<T>(normalizePerType<T>(imgData), ret_val);

    return ret_val;
}

af_err af_draw_image(const af_window window,
                     const af_array in, const af_cell* const props)
{
    if(window == 0) {
        fprintf(stderr, "Not a valid window\n");
        return AF_SUCCESS;
    }
    try {
        const ArrayInfo& info = getInfo(in);

        af::dim4 in_dims = info.dims();
        af_dtype type    = info.getType();
        DIM_ASSERT(0, in_dims[2] == 1 || in_dims[2] == 3 || in_dims[2] == 4);
        DIM_ASSERT(0, in_dims[3] == 1);

        makeContextCurrent(window);
        fg_image image = NULL;

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
        FG_CHECK(fg_set_window_colormap(window, (fg_color_map)props->cmap));
        if (props->col>-1 && props->row>-1) {
            FG_CHECK(fg_draw_image_to_cell(window,
                                           gridDims.first, gridDims.second,
                                           props->row * gridDims.second + props->col,
                                           image, props->title, true));
        } else {
            FG_CHECK(fg_draw_image(window, image, true));
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}
