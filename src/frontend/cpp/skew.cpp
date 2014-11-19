/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/image.h>
#include "error.hpp"

namespace af
{

array skew(const array& in, const float skew0, const float skew1, const dim_type odim0, const dim_type odim1, const bool inverse)
{
    af_array out = 0;
    AF_THROW(af_skew(&out, in.get(), skew0, skew1, odim0, odim1, inverse));
    return array(out);
}

}
