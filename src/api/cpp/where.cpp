/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/algorithm.h>
#include <af/gfor.h>
#include "error.hpp"

namespace af
{
    array where(const array& in)
    {
        if (gforGet()) {
            AF_THROW_MSG("WHERE can not be used inside GFOR", AF_ERR_RUNTIME);
        }

        af_array out = 0;
        AF_THROW(af_where(&out, in.get()));
        return array(out);
    }
}
