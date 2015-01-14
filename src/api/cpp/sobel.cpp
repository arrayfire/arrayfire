/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/image.h>
#include <af/arith.h>
#include "error.hpp"

namespace af
{

void sobel(array &dx, array &dy, const array &img, const unsigned ker_size=3)
{
    af_array af_dx = 0;
    af_array af_dy = 0;
    AF_THROW(af_sobel_dxdy(&af_dx, &af_dy, img.get(), ker_size));
    dx = array(af_dx);
    dy = array(af_dy);
}

array sobel(const array &img, const unsigned ker_size=3, bool isFast=false)
{
    array dx;
    array dy;
    sobel(dx, dy, img, ker_size);
    if (isFast) {
        return abs(dx)+abs(dy);
    } else {
        return sqrt(dx*dx+dy*dy);
    }
}

}

