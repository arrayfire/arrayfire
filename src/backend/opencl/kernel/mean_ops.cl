/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

To transform(Ti in) { return (To)(in); }

void binOp(To *lhs, Tw *l_wt, To rhs, Tw r_wt) {
    Tw l_scale = (*l_wt);
    (*l_wt) += (r_wt);
    if (((*l_wt) != 0) || ((r_wt) != 0)) {
        l_scale = l_scale / (*l_wt);

        Tw r_scale = (r_wt) / (*l_wt);
        (*lhs)     = (l_scale * (*lhs)) + (r_scale * (rhs));
    }
}

//
// Since only 1 pass is used, the rounding errors will become important because
// the longer the serie the larger the difference between the 2 numbers to sum
// See: https://en.wikipedia.org/wiki/Kahan_summation_algorithm to reduce this
// error
void binOpWithCorr(To *c, To *lhs, Tw *l_wt, To rhs, Tw r_wt) {
    (*l_wt) += (r_wt);
    if (((*l_wt) != 0) || ((r_wt) != 0)) {
        // (*lhs) = (*lhs) + ((rhs) - (*lhs)) * ((r_wt) / (*l_wt));

        // *c is zero for the first time around
        To y = ((rhs) - (*lhs)) * ((r_wt) / (*l_wt)) - (*c);
        // Alas, (*lhs) is big, y small, so low-order digits of y are lost
        To t = (*lhs) + y;
        // (t - (*lhs)) cancels the high-order part of y
        // subtracting y recovers negative (low part of y)
        (*c) = (t - (*lhs)) - y;
        // Algebraically, *c should always be zero.  Beware overly-agressive
        // optimizing compilers!
        (*lhs) = t;
        // Next time around, the lost low part will be added to y in a fresh
        // attempt
    }
}
