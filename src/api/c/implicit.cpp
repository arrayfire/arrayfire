/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <implicit.hpp>

/*
Implicit type mimics C/C++ behavior.

Order of precedence:
- complex > real
- double > float > uint > int > uchar > char
*/

af_dtype implicit(const af_dtype lty, const af_dtype rty)
{
    if (lty == rty) {
        return lty;
    }

    if (lty == c64 || rty == c64) {
           return c64;
    }

    if (lty == c32 || rty == 32) {
        if (lty == f64 || rty == f64)  return c64;
        return c32;
    }

    if (lty == f64 || rty == f64) return f64;
    if (lty == f32 || rty == f32) return f32;

    if ((lty == u32) ||
        (rty == u32)) return u32;

    if ((lty == s32) ||
        (rty == s32)) return s32;

    if ((lty == u8 ) ||
        (rty == u8 )) return u8;

    if ((lty == b8 ) &&
        (rty == b8 )) return b8;

    return f32;
}

af_dtype implicit(const af_array lhs, const af_array rhs)
{
    ArrayInfo lInfo = getInfo(lhs);
    ArrayInfo rInfo = getInfo(rhs);

    return implicit(lInfo.getType(), rInfo.getType());
}

af_array cast(const af_array in, const af_dtype type)
{
    const ArrayInfo info = getInfo(in);

    if (info.getType() == type) {
        return weakCopy(in);
    }

    switch (type) {
    case f32: return cast<float   >(in);
    case f64: return cast<double  >(in);
    case c32: return cast<cfloat  >(in);
    case c64: return cast<cdouble >(in);
    case s32: return cast<int     >(in);
    case u32: return cast<uint    >(in);
    case u8 : return cast<uchar   >(in);
    case b8 : return cast<char    >(in);
    default: TYPE_ERROR(2, type);
    }
}
