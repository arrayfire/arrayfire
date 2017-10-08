/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <common/defines.hpp>
#include <math.hpp>

namespace cpu
{

uint abs(uint val) { return val; }
uchar abs(uchar val) { return val; }
uintl abs(uintl val) { return val; }

cfloat  scalar(float val)
{
    cfloat  cval = {(float)val, 0};
    return cval;
}

cdouble scalar(double val)
{
    cdouble  cval = {val, 0};
    return cval;
}

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
