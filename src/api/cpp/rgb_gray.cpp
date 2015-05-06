/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/image.h>
#include <af/array.h>
#include "error.hpp"

namespace af
{

array rgb2gray(const array& in, const float rPercent, const float gPercent, const float bPercent)
{
    af_array temp = 0;
    AF_THROW(af_rgb2gray(&temp, in.get(), rPercent, gPercent, bPercent));
    return array(temp);
}

array gray2rgb(const array& in, const float rFactor, const float gFactor, const float bFactor)
{
    af_array temp = 0;
    AF_THROW(af_gray2rgb(&temp, in.get(), rFactor, gFactor, bFactor));
    return array(temp);
}

}
