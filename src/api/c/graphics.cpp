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

af_err af_image(int *out, const af_array in, const int wId, const char *title)
{
    try {
        ArrayInfo info = getInfo(in);

        af::dim4 in_dims = info.dims();
        DIM_ASSERT(1, in_dims[2] == 1 || in_dims[2] == 3 || in_dims[2] == 4);
        DIM_ASSERT(1, in_dims[3] == 1);
        ARG_ASSERT(1, info.getType() == f32);

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

        int output = image(getArray<float>(X), wId, title);

        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
