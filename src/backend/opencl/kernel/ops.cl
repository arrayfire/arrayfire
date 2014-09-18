#if T == double || Ti == double || To == double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

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
    return in;
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
    return in;
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
