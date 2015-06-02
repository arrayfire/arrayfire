/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/image.h>
#include <af/array.h>
#include "error.hpp"

namespace af
{

array scale(const array& in, const float scale0, const float scale1, const dim_t odim0, const dim_t odim1, const interpType method)
{
    af_array out = 0;
    AF_THROW(af_scale(&out, in.get(), scale0, scale1, odim0, odim1, method));
    return array(out);
}

}
