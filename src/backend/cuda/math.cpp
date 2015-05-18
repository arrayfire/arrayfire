/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <math.hpp>

namespace cuda
{
    cfloat division(cfloat lhs, double rhs)
    {
        cfloat retVal;
        retVal.x = real(lhs) / rhs;
        retVal.y = imag(lhs) / rhs;
        return retVal;
    }

    cdouble division(cdouble lhs, double rhs)
    {
        cdouble retVal;
        retVal.x = real(lhs) / rhs;
        retVal.y = imag(lhs) / rhs;
        return retVal;
    }
}
