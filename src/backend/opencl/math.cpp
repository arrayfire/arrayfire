/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "math.hpp"

namespace opencl
{
    bool operator ==(cfloat a, cfloat b) { return (a.s[0] == b.s[0]) && (a.s[1] == b.s[1]); }
    bool operator !=(cfloat a, cfloat b) { return !(a == b); }
    bool operator ==(cdouble a, cdouble b) { return (a.s[0] == b.s[0]) && (a.s[1] == b.s[1]); }
    bool operator !=(cdouble a, cdouble b) { return !(a == b); }

    cfloat operator +(cfloat a, cfloat b)
    {
        cfloat res = {{a.s[0] + b.s[0], a.s[1] + b.s[1]}};
        return res;
    }

    cdouble operator +(cdouble a, cdouble b)
    {
        cdouble res = {{a.s[0] + b.s[0], a.s[1] + b.s[1]}};
        return res;
    }

    cfloat operator *(cfloat lhs, cfloat rhs)
    {
        cfloat out;
        out.s[0] = lhs.s[0] * rhs.s[0] - lhs.s[1] * rhs.s[1];
        out.s[1] = lhs.s[0] * rhs.s[1] + lhs.s[1] * rhs.s[0];
        return out;
    }

    cdouble operator *(cdouble lhs, cdouble rhs)
    {
        cdouble out;
        out.s[0] = lhs.s[0] * rhs.s[0] - lhs.s[1] * rhs.s[1];
        out.s[1] = lhs.s[0] * rhs.s[1] + lhs.s[1] * rhs.s[0];
        return out;
    }

    cfloat division(cfloat lhs, double rhs)
    {
        cfloat retVal;
        retVal.s[0] = real(lhs) / rhs;
        retVal.s[1] = imag(lhs) / rhs;
        return retVal;
    }

    cdouble division(cdouble lhs, double rhs)
    {
        cdouble retVal;
        retVal.s[0] = real(lhs) / rhs;
        retVal.s[1] = imag(lhs) / rhs;
        return retVal;
    }
}
