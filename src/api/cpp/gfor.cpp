/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/seq.h>
#include <af/array.h>
#include <af/gfor.h>
#include "error.hpp"

namespace af
{

    static bool gforStatus;

    bool gforGet() { return gforStatus; }
    void gforSet(bool val) { gforStatus = val; }

    bool gforToggle()
    {
        bool status = gforGet();
        status ^= 1;
        gforSet(status);
        return status;
    }

    array batchFunc(const array &lhs, const array &rhs, batchFunc_t func)
    {
        if (gforGet()) AF_THROW_MSG("batchFunc can not be used inside GFOR",
                                    AF_ERR_ARG);
        gforSet(true);
        array res = func(lhs, rhs);
        gforSet(false);
        return res;
    }

}
