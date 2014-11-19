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

array translate(const array& in, const float trans0, const float trans1, const dim_type odim0, const dim_type odim1)
{
    af_array out = 0;
    AF_THROW(af_translate(&out, in.get(), trans0, trans1, odim0, odim1));
    return array(out);
}

}
