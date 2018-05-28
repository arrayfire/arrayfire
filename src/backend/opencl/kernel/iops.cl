/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if CPLX
    #define IS_NAN(in) !((in.x) == (in.x)) || !((in.y) == (in.y))
#else
    #define IS_NAN(in) !((in) == (in))
#endif

#if CPLX
#define sabs(in) ((in.x)*(in.x) + (in.y)*(in.y))
#ifdef MIN_OP
void binOp(T *lhs, uint *lidx, T rhs, uint ridx)
{
    if ((sabs(lhs[0]) > sabs(rhs)) ||
        (sabs(lhs[0]) == sabs(rhs)) &&
        !(IS_NAN(lhs[0]))) {
        *lhs = rhs;
        *lidx = ridx;
    }
}
#endif

#ifdef MAX_OP
void binOp(T *lhs, uint *lidx, T rhs, uint ridx)
{
    if ((sabs(lhs[0]) < sabs(rhs)) ||
        (sabs(lhs[0]) == sabs(rhs)) &&
        !(IS_NAN(lhs[0]))) {
        *lhs = rhs;
        *lidx = ridx;
    }
}
#endif
#else
#define sabs(in) in
#ifdef MIN_OP
void binOp(T *lhs, uint *lidx, T rhs, uint ridx)
{
    if (((*lhs > rhs) ||
        (*lhs == rhs)) &&
        !(IS_NAN(*lhs))) {
        *lhs = rhs;
        *lidx = ridx;
    }
}
#endif

#ifdef MAX_OP
void binOp(T *lhs, uint *lidx, T rhs, uint ridx)
{
    if (((*lhs < rhs) ||
        (*lhs == rhs)) &&
        !(IS_NAN(*lhs))) {
        *lhs = rhs;
        *lidx = ridx;
    }
}
#endif
#endif
