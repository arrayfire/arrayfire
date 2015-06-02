/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/image.h>
#include <af/compatible.h>
#include <af/array.h>
#include "error.hpp"

namespace af
{

array colorspace(const array& image, const CSpace to, const CSpace from)
{
    return colorSpace(image, to, from);
}

array colorSpace(const array& image, const CSpace to, const CSpace from)
{
    af_array temp = 0;
    AF_THROW(af_color_space(&temp, image.get(), to ,from));
    return array(temp);
}

}
