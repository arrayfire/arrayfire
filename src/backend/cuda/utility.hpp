/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include "backend.hpp"

namespace cuda
{

static __DH__ dim_t trimIndex(const int &idx, const dim_t &len)
{
    int ret_val = idx;
    if (ret_val<0) {
        int offset  = (abs(ret_val)-1)%len;
        ret_val = offset;
    } else if (ret_val>=len) {
        int offset  = abs(ret_val)%len;
        ret_val = len-offset-1;
    }
    return ret_val;
}

}
