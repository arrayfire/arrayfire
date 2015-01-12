/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/graphics.h>
#include "error.hpp"

namespace af
{
    int image(const array &in, const int wId, const char *title, const float scale)
    {
        return image(in, scale, scale, wId, title);
    }

    int image(const array &in, const float scale_w, const float scale_h, const int wId, const char *title)
    {
        int out = 0;
        AF_THROW(af_image_s(&out, in.get(), wId, title, scale_w, scale_h));
        return out;
    }

    int image(const array &in, const dim_type disp_w, const dim_type disp_h, const int wId, const char *title)
    {
        int out = 0;
        AF_THROW(af_image_d(&out, in.get(), wId, title, disp_w, disp_h));
        return out;
    }
}
