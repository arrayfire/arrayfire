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

array bilateral(const array &in, const float spatial_sigma, const float chromatic_sigma, const bool is_color)
{
    af_array out = 0;
    AF_THROW(af_bilateral(&out, in.get(), spatial_sigma, chromatic_sigma, is_color));
    return array(out);
}

}
