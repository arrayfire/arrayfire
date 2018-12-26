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
    if (((*l_wt) != 0) || (r_wt != 0)) {
        Tw l_scale = (*l_wt);
        (*l_wt) += r_wt;
        l_scale = l_scale / (*l_wt);

        Tw r_scale = r_wt / (*l_wt);
        (*lhs)     = (l_scale * (*lhs)) + (r_scale * rhs);
    }
}
