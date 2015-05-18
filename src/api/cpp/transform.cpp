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

array transform(const array& in, const array& transform, const dim_t odim0, const dim_t odim1, const interpType method, const bool inverse)
{
    af_array out = 0;
    AF_THROW(af_transform(&out, in.get(), transform.get(), odim0, odim1, method, inverse));
    return array(out);
}

}
