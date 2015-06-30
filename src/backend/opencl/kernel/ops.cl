/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define IS_NAN(in) !((in) == (in))

#ifdef ADD_OP
T binOp(T lhs, T rhs)
{
    return lhs + rhs;
}

To transform(Ti in)
{
    return(To)(in);
}
#endif

#ifdef MUL_OP
#if CPLX
T binOp(T lhs, T rhs)
{
    T out;
    out.x = lhs.x * rhs.x - lhs.y * rhs.y;
    out.y = lhs.x * rhs.y + lhs.y * rhs.x;
    return out;
}
#else
T binOp(T lhs, T rhs)
{
    return lhs * rhs;
}
#endif

To transform(Ti in)
{
    return(To)(in);
}
#endif

#ifdef OR_OP
uchar binOp(uchar lhs, uchar rhs)
{
    return lhs || rhs;
}

#if CPLX
uchar transform(Ti in)
{
    return (in.x != 0) || (in.y != 0);
}
#else
uchar transform(Ti in)
{
    return (in != 0);
}
#endif
#endif

#ifdef AND_OP
uchar binOp(uchar lhs, uchar rhs)
{
    return lhs && rhs;
}

#if CPLX
uchar transform(Ti in)
{
    return (in.x != 0) || (in.y != 0);
}
#else
uchar transform(Ti in)
{
    return (in != 0);
}
#endif
#endif

#ifdef NOTZERO_OP
uint binOp(uint lhs, uint rhs)
{
    return lhs + rhs;
}

#if CPLX
uint transform(Ti in)
{
    return (in.x != 0) || (in.y != 0);
}
#else
uint transform(Ti in)
{
    return (in != 0);
}
#endif
#endif

#ifdef MIN_OP

T transform(T in)
{
    T val = init;
    return IS_NAN(in) ? (val) : (in);
}

#if CPLX
#define sabs(in) ((in.x)*(in.x) + (in.y)*(in.y))
#else
#define sabs(in) in
#endif

T binOp(T lhs, T rhs)
{
    return sabs(lhs) < sabs(rhs) ? lhs : rhs;
}
#endif

#ifdef MAX_OP

T transform(T in)
{
    T val = init;
    return IS_NAN(in) ? (val) : (in);
}

#if CPLX
#define sabs(in) ((in.x)*(in.x) + (in.y)*(in.y))
#else
#define sabs(in) in
#endif

T binOp(T lhs, T rhs)
{
    return sabs(lhs) > sabs(rhs) ? lhs : rhs;
}
#endif
