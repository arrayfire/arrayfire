/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void example(__global T *       d_dst,
             KParam             oInfo,
             __global const T * d_src,
             KParam             iInfo,
             int                method);
{
    // kernel algorithm goes here
}
