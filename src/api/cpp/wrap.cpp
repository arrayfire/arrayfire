/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/image.h>
#include "error.hpp"

namespace af
{
    array wrap(const array& in,
               const dim_t ox, const dim_t oy,
               const dim_t wx, const dim_t wy,
               const dim_t sx, const dim_t sy,
               const dim_t px, const dim_t py,
               const bool is_column)
    {
        af_array out = 0;
        AF_THROW(af_wrap(&out, in.get(), ox, oy, wx, wy, sx, sy, px, py, is_column));
        return array(out);
    }
}
