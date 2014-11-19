/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/blas.h>
#include "error.hpp"

namespace af
{

array transpose(const array& in)
{
    af_array out = 0;
    AF_THROW(af_transpose(&out, in.get()));
    return array(out);
}

}
