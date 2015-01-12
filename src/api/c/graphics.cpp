/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/graphics.h>
#include <af/index.h>
#include <graphics.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>

using af::dim4;
using namespace detail;

af_err af_image_s(int *out, const af_array in, const int wId, const char *title,
                  const float scale_w, const float scale_h)
{
    try {
        ArrayInfo info = getInfo(in);

        af::dim4 in_dims = info.dims();
        DIM_ASSERT(1, in_dims[2] == 1 || in_dims[2] == 3 || in_dims[2] == 4);
        DIM_ASSERT(1, in_dims[3] == 1);
        ARG_ASSERT(1, info.getType() == f32);
        ARG_ASSERT(4, scale_w > 0.0f);
        ARG_ASSERT(5, scale_h > 0.0f);

        // Tile if needed
        // Interleave values and transpose
        af_array X, Y;
        if(in_dims[2] == 1) {
            af_tile(&Y, in, 1, 1, 3, 1);
            af_reorder(&X, Y, 2, 1, 0, 3);
        } else if (in_dims[2] == 4) {
            //FIXME
            //Y = in(span, span, seq(2));
        } else {
            af_reorder(&X, in, 2, 1, 0, 3);
        }

        int output = image(getArray<float>(X), wId, title, in_dims[1] * scale_w, in_dims[0] * scale_h);

        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_image_d(int *out, const af_array in, const int wId, const char *title,
                  const dim_type disp_w, const dim_type disp_h)
{
    try {
        ArrayInfo info = getInfo(in);

        af::dim4 in_dims = info.dims();
        DIM_ASSERT(1, in_dims[2] == 1 || in_dims[2] == 3 || in_dims[2] == 4);
        DIM_ASSERT(1, in_dims[3] == 1);
        ARG_ASSERT(1, info.getType() == f32);
        ARG_ASSERT(4, disp_w > 0 || disp_w == -1);
        ARG_ASSERT(5, disp_h > 0 || disp_h == -1);

        // Tile if needed
        // Interleave values and transpose
        af_array X, Y;
        if(in_dims[2] == 1) {
            af_tile(&Y, in, 1, 1, 3, 1);
            af_reorder(&X, Y, 2, 1, 0, 3);
        } else if (in_dims[2] == 4) {
            //FIXME
            //Y = in(span, span, seq(2));
        } else {
            af_reorder(&X, in, 2, 1, 0, 3);
        }

        dim_type dw = disp_w, dh = disp_h;
        if(dw == -1)
            dw = in_dims[1];
        if(dh == -1)
            dh = in_dims[0];

        int output = image(getArray<float>(X), wId, title, dw, dh);

        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
