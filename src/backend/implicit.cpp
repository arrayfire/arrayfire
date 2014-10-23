#include <implicit.hpp>

/*
Implicit type mimics C/C++ behavior.

Order of precedence:
- complex > real
- double > float > uint > int > uchar > char
*/

af_dtype implicit(const af_array lhs, const af_array rhs)
{
    ArrayInfo lInfo = getInfo(lhs);
    ArrayInfo rInfo = getInfo(rhs);

    if (lInfo.getType() == rInfo.getType()) {
        return lInfo.getType();
    }

    if (lInfo.isComplex() || rInfo.isComplex()) {
        if (lInfo.isDouble() || rInfo.isDouble()) return c64;
        return c32;
    }

    if (lInfo.isDouble() || rInfo.isDouble()) return f64;
    if (lInfo.isSingle() || rInfo.isSingle()) return f32;

    af_dtype ltype = lInfo.getType();
    af_dtype rtype = lInfo.getType();

    if ((ltype == u32) ||
        (rtype == u32)) return u32;

    if ((ltype == s32) ||
        (rtype == s32)) return s32;

    if ((ltype == u8 ) ||
        (rtype == u8 )) return u8;

    if ((ltype == s8 ) ||
        (rtype == s8 )) return s8;

    if ((ltype == b8 ) &&
        (rtype == b8 )) return b8;

    return f32;
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
    case s8 : return cast<char    >(in);
    case u8 : return cast<uchar   >(in);
    case b8 : return cast<uchar   >(in);
    default: TYPE_ERROR(2, type);
    }
}
