/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel
void
binaryOp(   global  R*      out,
            global  T*      lhs,
            global  U*      rhs,
            const   unsigned long elements)
{
    size_t idx = get_global_id(0);
    if(idx < elements) {
        out[idx] = lhs[idx] OP rhs[idx];

    }
}

