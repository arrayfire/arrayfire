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

    array hsv2rgb(const array& in)
    {
        af_array temp = 0;
        AF_THROW(af_hsv2rgb(&temp, in.get()));
        return array(temp);
    }

    array rgb2hsv(const array& in)
    {
        af_array temp = 0;
        AF_THROW(af_rgb2hsv(&temp, in.get()));
        return array(temp);
    }

}
