/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if CPLX
#define sabs(in) ((in.x)*(in.x) + (in.y)*(in.y))
#else
#define sabs(in) in
#endif

#ifdef MIN_OP
void binOp(T *lhs, uint *lidx, T rhs, uint ridx)
{
    if ((*lhs > rhs) ||
        (*lhs == rhs && *lidx < ridx)) {
        *lhs = rhs;
        *lidx = ridx;
    }
}
#endif

#ifdef MAX_OP
void binOp(T *lhs, uint *lidx, T rhs, uint ridx)
{
    if ((*lhs < rhs) ||
        (*lhs == rhs && *lidx > ridx)) {
        *lhs = rhs;
        *lidx = ridx;
    }
}
#endif
