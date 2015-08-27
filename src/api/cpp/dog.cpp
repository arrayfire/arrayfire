/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/vision.h>
#include "error.hpp"

namespace af
{

array dog(const array& in, const int radius1, const int radius2)
{
    af_array temp = 0;
    AF_THROW(af_dog(&temp, in.get(), radius1, radius2));
    return array(temp);
}

}
