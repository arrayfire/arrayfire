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

array resize(const array &in, const dim_type odim0, const dim_type odim1, const af_interp_type method)
{
    af_array out = 0;
    AF_THROW(af_resize(&out, in.get(), odim0, odim1, method));
    return array(out);
}

}
