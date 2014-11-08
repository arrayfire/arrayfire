/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <math.hpp>

namespace cpu
{
    cfloat min(cfloat lhs, cfloat rhs)
    {
        return abs(lhs) < abs(rhs) ? lhs : rhs;
    }

    cdouble min(cdouble lhs, cdouble rhs)
    {
        return abs(lhs) < abs(rhs) ? lhs : rhs;
    }

    cfloat max(cfloat lhs, cfloat rhs)
    {
        return abs(lhs) > abs(rhs) ? lhs : rhs;
    }

    cdouble max(cdouble lhs, cdouble rhs)
    {
        return abs(lhs) > abs(rhs) ? lhs : rhs;
    }
}
