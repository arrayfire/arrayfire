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
    int image(const array &in, const int wId, const char *title)
    {
        int out = 0;
        AF_THROW(af_image(&out, in.get(), wId, title));
        return out;
    }
}
